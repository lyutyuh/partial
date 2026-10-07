"""Order-theoretic arc aggregation for any order dimension K in O(N log^(K-1) N), without the (N, N) score matrix.

Scores follow Eq. 2 of Liu et al. (EMNLP 2023) with K realizer values per side:

    s(x, y) = -F(x, y),    F(x, y) = max_k (f_k(x) - g_k(y)).

Head y falls in branch k of query x when f_k(x) - g_k(y) is the largest difference (ties go to the smallest k), i.e.

    g_j(y) - g_k(y) >= f_j(x) - f_k(x)   for all j != k   (strict for j < k).

On branch k the score factorises, s = g_k(y) - f_k(x), so the per-query aggregation becomes K dominance queries in K - 1
dimensions over the keys (u_j(y) = g_j(y) - g_k(y)) with log-weights g_k(y). Each one is answered by a range tree built on
dyadic blocks: sort the heads by the first coordinate, split a query's dominated prefix into at most log N aligned blocks,
and recurse on the remaining coordinates inside each block. Every level is a batched ``sort`` + ``searchsorted`` and the
innermost level a ``logcumsumexp`` (training) or ``cummax`` (decoding), so the whole thing is ordinary tensor code: exact,
batched over sentences, differentiable through autograd, and O(N log^(K-1) N) time (one log factor above the paper's
App. E.2 conjecture, from re-sorting at each level) with O(N log^(K-2) N) memory.

The K = 2 case reduces to a sort and two scans and has a dedicated Triton kernel in ``linear_order``; this module is the
general-K path (K = 3 is 2-D dominance, K = 4 is 3-D, ...).
"""
import math

import torch
import torch.nn.functional as F

NEG = float("-inf")


def _sortkey(x):
    """Order-preserving float32 -> int64 map into [0, 2^32); -0.0 is canonicalised to +0.0."""
    x = x.float() + 0.0
    b = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(b >= 0x80000000, (~b) & 0xFFFFFFFF, b | 0x80000000)


def _composite(seg, coord):
    """Key that sorts by segment ascending, then by coordinate DESCENDING (so a dominated set is a prefix)."""
    return (seg << 32) | _sortkey(-coord)


class _LogSumExp:
    @staticmethod
    def prefix(v, idx):
        return torch.logcumsumexp(v, dim=-1), None

    @staticmethod
    def combine(vals, idxs):
        """Aggregate along the last dim."""
        return torch.logsumexp(vals, dim=-1), None


class _Max:
    @staticmethod
    def prefix(v, idx):
        m, i = torch.cummax(v, dim=-1)
        return m, idx.gather(-1, i)

    @staticmethod
    def combine(vals, idxs):
        best = vals.argmax(dim=-1, keepdim=True)
        return vals.gather(-1, best).squeeze(-1), idxs.gather(-1, best).squeeze(-1)


def _dominance(hcoords, hw, hidx, qcoords, strict, op):
    """Aggregate hw over the heads whose coordinates all dominate the query's: coord_d >= qcoord_d (> where strict).

    Rows are independent problems (sentence x branch). Heads: coordinates ``hcoords`` (list of D tensors (R, M), M a
    power of two), log-weights ``hw`` (R, M), identities ``hidx`` (R, M) or None. Queries: ``qcoords`` (list of D
    tensors (R, Q)); ``strict`` is a (R, D) bool tensor. Returns (values (R, Q), identities or None).

    Range tree on dyadic blocks, one depth per coordinate. At depth d the heads of every level tuple live in one
    flattened array of T * M items sorted by (tuple, block, -coord_d), so each depth is a single sort and a single
    searchsorted for all branches, levels and sentences; a query's dominated prefix inside its block splits into one
    aligned block of size 2^l per set bit l of its length, which become the query's copies at the next depth.
    """
    R, M = hw.shape
    Q = qcoords[0].shape[1]
    D = len(hcoords)
    dev = hw.device
    levels = int(math.log2(M)) + 1
    ar = torch.arange(M, device=dev)
    sizes = [M]  # block size of each tuple
    h_seg = torch.zeros(R, M, dtype=torch.int64, device=dev)  # seg = tuple * M + block
    q_seg = torch.zeros(R, Q, dtype=torch.int64, device=dev)
    q_ok = torch.ones(R, Q, dtype=torch.bool, device=dev)
    hvals, hid, coords, qc = hw, hidx, list(hcoords), list(qcoords)
    for d in range(D):
        key, perm = ((h_seg << 32) | _sortkey(-coords[0])).sort(dim=-1)  # coords holds the not-yet-used dims
        hvals = hvals.gather(-1, perm)
        hid = hid.gather(-1, perm) if hid is not None else None
        coords = [c.gather(-1, perm) for c in coords[1:]]
        T = len(sizes)
        size_t = torch.tensor(sizes, device=dev)
        qkey = ((q_seg << 32) | _sortkey(-qc[d])) - strict[:, d:d + 1].to(torch.int64)  # keys are injective ints
        r = torch.searchsorted(key, qkey, right=True)  # end of the dominated prefix in the flattened array
        qt, qb = q_seg // M, q_seg % M
        qstart = qt * M + qb * size_t[qt]
        n = r - qstart  # prefix length inside the query's block
        if d == D - 1:
            parts_v, parts_i = [], []
            for t, S in enumerate(sizes):
                sl = slice(t * M, (t + 1) * M)
                P, I = op.prefix(hvals[:, sl].reshape(R, -1, S), hid[:, sl].reshape(R, -1, S) if hid is not None else None)
                parts_v.append(P.reshape(R, M))
                parts_i.append(I.reshape(R, M) if I is not None else None)
            P = torch.cat(parts_v, dim=-1)
            pos = (r - 1).clamp(min=0)
            val = torch.where(q_ok & (n > 0), P.gather(-1, pos), NEG).view(R, Q, -1)
            idx = torch.cat(parts_i, dim=-1).gather(-1, pos).view(R, Q, -1) if hid is not None else None
            return op.combine(val, idx)
        # heads: tuple (t, l) for every level l <= log2(size_t); block = position within t's array >> l
        lut = torch.full((T, levels), -1, dtype=torch.int64, device=dev)
        new_sizes, src, block = [], [], []
        for t, S in enumerate(sizes):
            for l in range(int(math.log2(S)) + 1):
                lut[t, l] = len(new_sizes)
                new_sizes.append(1 << l)
                src.append(t * M + ar)
                block.append(ar >> l)
        src = torch.cat(src)
        h_seg = (torch.repeat_interleave(torch.arange(len(new_sizes), device=dev), M) * M + torch.cat(block)).expand(R, -1)
        hvals = hvals[:, src]
        hid = hid[:, src] if hid is not None else None
        coords = [c[:, src] for c in coords]
        sizes = new_sizes
        # queries: one copy per level l; alive when bit l of n is set (and the level exists for its tuple)
        lv = torch.arange(levels, device=dev)
        n_e = n.unsqueeze(-1)
        has = ((n_e >> lv) & 1) == 1
        new_qt = lut[qt.unsqueeze(-1).expand(-1, -1, levels), lv.expand(R, q_seg.shape[1], -1)]
        start_in_t = (qstart - qt * M).unsqueeze(-1) + ((n_e >> (lv + 1)) << (lv + 1))
        new_qb = torch.minimum(start_in_t >> lv, (M >> lv) - 1)
        q_ok = (q_ok.unsqueeze(-1) & has & (new_qt >= 0)).reshape(R, -1)
        q_seg = (new_qt.clamp(min=0) * M + new_qb).reshape(R, -1)
        qc = [c.unsqueeze(-1).expand(-1, -1, levels).reshape(R, -1) for c in qc]
    raise AssertionError("unreachable")


def _prepare(f, g, lengths):
    if f.shape != g.shape or f.dim() != 3:
        raise ValueError(f"expected f, g of shape (B, L, K), got {tuple(f.shape)} and {tuple(g.shape)}")
    B, L, K = f.shape
    S = 1 << max(L - 1, 0).bit_length()  # pad to a power of two so every dyadic block is full
    f = F.pad(f.float(), (0, 0, 0, S - L))
    g = F.pad(g.float(), (0, 0, 0, S - L))
    lengths = lengths.to(f.device)
    if int(lengths.max()) > L:
        raise ValueError(f"lengths up to {int(lengths.max())} exceed the padded length {L}")
    valid = torch.arange(S, device=f.device)[None, :] < lengths[:, None]
    return f, g, valid, S, L


def _branches(f, g, valid, S, op, with_idx):
    """Per-branch aggregates (value - f_k), all K branches in one fused call; returns (K, B, S) values and identities."""
    B, _, K = f.shape
    if K == 1:  # every head is on the only branch
        w = torch.where(valid, g[..., 0], NEG)
        if with_idx:
            v, i = w.max(dim=-1, keepdim=True)
            return (v - f[..., 0]).unsqueeze(0), i.expand(B, S).unsqueeze(0)
        return (torch.logsumexp(w, dim=-1, keepdim=True) - f[..., 0]).unsqueeze(0), None
    hcoords, qcoords, strict = [], [], []
    for d in range(K - 1):
        hc, qcd, st = [], [], []
        for k in range(K):
            j = [j for j in range(K) if j != k][d]
            hc.append(torch.where(valid, g[..., j] - g[..., k], NEG))
            qcd.append(f[..., j] - f[..., k])
            st.append(torch.full((B,), j < k, dtype=torch.bool, device=f.device))
        hcoords.append(torch.cat(hc, 0))
        qcoords.append(torch.cat(qcd, 0))
        strict.append(torch.cat(st, 0))
    w = torch.cat([torch.where(valid, g[..., k], NEG) for k in range(K)], 0)
    hidx = torch.arange(S, device=f.device).expand(K * B, S) if with_idx else None
    v, i = _dominance(hcoords, w, hidx, qcoords, torch.stack(strict, -1), op)
    fk = f.permute(2, 0, 1)  # (K, B, S)
    return v.view(K, B, S) - fk, (i.view(K, B, S) if i is not None else None)


def _branch_inputs(f, g, valid, K):
    """Dominance inputs for all K branches stacked along rows: coords (K-1 lists of (K*B, S)), weights, strict."""
    B, S, _ = f.shape
    hcoords, qcoords, strict = [], [], []
    for d in range(K - 1):
        hc, qcd, st = [], [], []
        for k in range(K):
            j = [j for j in range(K) if j != k][d]
            hc.append(torch.where(valid, g[..., j] - g[..., k], NEG))
            qcd.append(f[..., j] - f[..., k])
            st.append(torch.full((B,), j < k, dtype=torch.bool, device=f.device))
        hcoords.append(torch.cat(hc, 0))
        qcoords.append(torch.cat(qcd, 0))
        strict.append(torch.cat(st, 0))
    return hcoords, qcoords, torch.stack(strict, -1)


class _LogPartitionK(torch.autograd.Function):
    """Z with a hand-written backward: dg is the transposed dominance problem, run as one more fused pass."""

    @staticmethod
    def forward(ctx, f, g, valid, include_root):
        with torch.no_grad():
            vals, _ = _branches(f, g, valid, f.shape[1], _LogSumExp, with_idx=False)  # (K, B, S): L_k - f_k
            if include_root:
                vals = torch.cat([vals, torch.zeros_like(vals[:1])], 0)
            z = torch.logsumexp(vals, 0)
        ctx.save_for_backward(f, g, valid, vals[: f.shape[-1]], z)
        return z

    @staticmethod
    def backward(ctx, gz):
        f, g, valid, branch, z = ctx.saved_tensors
        B, S, K = f.shape
        gz = torch.where(valid, gz, 0.0)
        p = torch.exp(branch - z)  # (K, B, S): mass of branch k for query x
        df = -(gz * p).permute(1, 2, 0)
        if K == 1:  # every head is in every query's set: dg(y) = exp(g(y)) * sum_x gz(x) exp(-f(x) - Z(x))
            w = torch.where(valid, g[..., 0], NEG)
            tot = (gz * torch.exp(-f[..., 0] - z)).sum(-1, keepdim=True)
            dg = (torch.exp(w) * tot).unsqueeze(-1)
            return df, torch.where(valid.unsqueeze(-1), dg, 0.0), None, None
        # transposed problem: "heads" are the queries x with log-weights log|gz(x)| - f_k(x) - Z(x) (split by sign),
        # "queries" are the heads y; y in D_k(x) <=> -v(x) dominates -u(y), with the same strictness pattern
        hcoords, qcoords, strict = _branch_inputs(f, g, valid, K)
        t_h = [torch.cat([-c, -c], 0) for c in qcoords]  # (2*K*B, S): negated query coords, as heads
        t_q = [torch.cat([-c, -c], 0) for c in hcoords]
        t_strict = torch.cat([strict, strict], 0)
        logw = []
        for sign in (1.0, -1.0):
            part = (gz * sign).clamp(min=0)
            lw = torch.where(valid & (part > 0), torch.log(part) - z, NEG)  # (B, S)
            logw.append(torch.cat([lw - f[..., k] for k in range(K)], 0))  # branch k rows
        t_w = torch.cat(logw, 0)
        T, _ = _dominance(t_h, t_w, None, t_q, t_strict, _LogSumExp)  # (2*K*B, S)
        T = T.view(2, K, B, S)
        gk = g.permute(2, 0, 1)  # (K, B, S)
        dg = (torch.exp(gk + T[0]) - torch.exp(gk + T[1])).permute(1, 2, 0)
        return df, torch.where(valid.unsqueeze(-1), dg, 0.0), None, None


def log_partition_k(f, g, lengths, include_root=True, autograd=True):
    """Z(x) = log sum_y exp(s(x, y)) for every dependent x, any K; differentiable in f and g.

    Args:
        f: Dependent realizer values, shape (B, L, K).
        g: Head realizer values, shape (B, L, K).
        lengths: Words per sentence, shape (B,); positions >= length are neither heads nor dependents.
        include_root: Add the ROOT head with score 0.
        autograd: Differentiate through the tree with autograd (default). ``False`` uses the hand-written backward
            (one transposed dominance pass with the upstream gradient split by sign); it is exact and equally fast
            but peaks at about twice the memory, because the forward tree itself, not autograd's saved tensors, sets
            the peak and the transposed pass doubles the rows.

    Returns:
        Z of shape (B, L), fp32, 0 at padding positions.
    """
    in_dtype = f.dtype
    f, g, valid, S, L = _prepare(f, g, lengths)
    if autograd:
        vals, _ = _branches(f, g, valid, S, _LogSumExp, with_idx=False)
        if include_root:
            vals = torch.cat([vals, torch.zeros_like(vals[:1])], 0)
        z = torch.logsumexp(vals, 0)
    else:
        z = _LogPartitionK.apply(f, g, valid, include_root)
    return torch.where(valid, z, 0.0)[:, :L]


@torch.no_grad()
def decode_k(f, g, lengths, include_root=True):
    """Greedy heads argmax_y s(x, y) for any K, in the repo convention (0 = ROOT, j + 1 = word j); 0 at padding."""
    f, g, valid, S, L = _prepare(f, g, lengths)
    vals, idxs = _branches(f, g, valid, S, _Max, with_idx=True)
    best, idx = _Max.combine(vals.movedim(0, -1), idxs.movedim(0, -1))
    heads = idx + 1
    if include_root:
        root = best <= 0.0  # torch.argmax picks the first column (ROOT) on ties
        best = torch.where(root, 0.0, best)
        heads = torch.where(root, 0, heads)
    heads = torch.where(valid, heads, 0)[:, :L].to(torch.int32)
    return heads, torch.where(valid, best, 0.0)[:, :L]


def gold_arc_score_k(f, g, heads):
    """s(x, h(x)) for gold heads (0 = ROOT, j + 1 = word j), with the kernels' branch rule at exact ties."""
    f = f.float()
    g = g.float()
    idx = (heads.clamp(min=1) - 1).long()
    diff = f - g.take_along_dim(idx.unsqueeze(-1), dim=1)  # (B, L, K): f_k(x) - g_k(h)
    k = diff.argmax(dim=-1, keepdim=True)  # first maximal k, the branch the aggregation assigns the arc to at a tie
    score = -diff.gather(-1, k).squeeze(-1)
    return torch.where(heads == 0, torch.zeros_like(score), score)


def arc_loss_k(f, g, heads, lengths, include_root=True):
    """Arc cross-entropy, mean over words, in O(N log^(K-1) N) memory; heads = -1 marks padding."""
    lengths = lengths.to(heads.device)
    valid = (heads >= 0) & (torch.arange(heads.shape[1], device=heads.device)[None, :] < lengths[:, None])
    nll = log_partition_k(f, g, lengths, include_root) - gold_arc_score_k(f, g, heads.clamp(min=0))
    return nll[valid].mean()
