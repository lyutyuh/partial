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
        return torch.logsumexp(torch.stack(vals, 0), 0), None


class _Max:
    @staticmethod
    def prefix(v, idx):
        m, i = torch.cummax(v, dim=-1)
        return m, idx.gather(-1, i)

    @staticmethod
    def combine(vals, idxs):
        v = torch.stack(vals, 0)
        best = v.argmax(0, keepdim=True)
        return v.gather(0, best).squeeze(0), torch.stack(idxs, 0).gather(0, best).squeeze(0)


def _dominance(hseg, S, hcoords, hw, hidx, qseg, qcoords, strict, op):
    """Aggregate hw over heads in the query's segment whose coordinates all dominate the query's.

    Heads: segment ids ``hseg`` (dense, every segment holds exactly ``S`` heads, S a power of two), coordinates
    ``hcoords`` (list of (B, M)), log-weights ``hw`` (B, M), identities ``hidx`` (B, M) or None. Queries: ``qseg``,
    ``qcoords`` (list of (B, Q)). ``strict[d]`` asks for coord > qcoord instead of >=. Returns (values, identities).
    """
    B, M = hw.shape
    key, perm = _composite(hseg, hcoords[0]).sort(dim=-1)
    hw = hw.gather(-1, perm)
    hidx = hidx.gather(-1, perm) if hidx is not None else None
    rest = [c.gather(-1, perm) for c in hcoords[1:]]
    r = torch.searchsorted(key, _composite(qseg, qcoords[0]), right=not strict[0])  # end of the dominated prefix
    n = r - qseg * S  # its length inside the query's segment, in [0, S]
    if len(hcoords) == 1:
        P, I = op.prefix(hw.view(B, -1, S), hidx.view(B, -1, S) if hidx is not None else None)
        pos = (r - 1).clamp(min=0)
        val = torch.where(n > 0, P.reshape(B, M).gather(-1, pos), NEG)
        return val, (I.reshape(B, M).gather(-1, pos) if I is not None else None)
    # split the prefix [seg*S, seg*S + n) into aligned dyadic blocks: one block of size 2^l per set bit l of n
    vals, idxs = [], []
    block_of = torch.arange(M, device=hw.device)
    for l in range(int(math.log2(S)) + 1):
        has = ((n >> l) & 1) == 1
        start = qseg * S + ((n >> (l + 1)) << (l + 1))
        new_qseg = (start >> l).clamp(max=(M >> l) - 1)
        v, i = _dominance((block_of >> l).expand(B, M), 1 << l, rest, hw, hidx, new_qseg, qcoords[1:], strict[1:], op)
        vals.append(torch.where(has, v, NEG))
        idxs.append(i)
    return op.combine(vals, idxs)


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
    """Per-branch aggregates (value - f_k) over the K branches; returns lists of (B, S) values and identities."""
    B, _, K = f.shape
    seg0 = torch.zeros(B, S, dtype=torch.int64, device=f.device)
    hidx = torch.arange(S, device=f.device).expand(B, S) if with_idx else None
    vals, idxs = [], []
    for k in range(K):
        others = [j for j in range(K) if j != k]
        w = torch.where(valid, g[..., k], NEG)
        if not others:  # K = 1: every head is on the only branch
            if with_idx:
                v, i = w.max(dim=-1, keepdim=True)
                v, i = v.expand(B, S), i.expand(B, S)
            else:
                v, i = torch.logsumexp(w, dim=-1, keepdim=True).expand(B, S), None
        else:
            hcoords = [torch.where(valid, g[..., j] - g[..., k], NEG) for j in others]
            qcoords = [f[..., j] - f[..., k] for j in others]
            v, i = _dominance(seg0, S, hcoords, w, hidx, seg0, qcoords, [j < k for j in others], op)
        vals.append(v - f[..., k])
        idxs.append(i)
    return vals, idxs


def log_partition_k(f, g, lengths, include_root=True):
    """Z(x) = log sum_y exp(s(x, y)) for every dependent x, any K; differentiable in f and g.

    Args:
        f: Dependent realizer values, shape (B, L, K).
        g: Head realizer values, shape (B, L, K).
        lengths: Words per sentence, shape (B,); positions >= length are neither heads nor dependents.
        include_root: Add the ROOT head with score 0.

    Returns:
        Z of shape (B, L), fp32, 0 at padding positions.
    """
    f, g, valid, S, L = _prepare(f, g, lengths)
    vals, _ = _branches(f, g, valid, S, _LogSumExp, with_idx=False)
    if include_root:
        vals.append(torch.zeros_like(vals[0]))
    z = torch.logsumexp(torch.stack(vals, 0), 0)
    return torch.where(valid, z, 0.0)[:, :L]


@torch.no_grad()
def decode_k(f, g, lengths, include_root=True):
    """Greedy heads argmax_y s(x, y) for any K, in the repo convention (0 = ROOT, j + 1 = word j); 0 at padding."""
    f, g, valid, S, L = _prepare(f, g, lengths)
    vals, idxs = _branches(f, g, valid, S, _Max, with_idx=True)
    best, idx = _Max.combine(vals, idxs)
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
