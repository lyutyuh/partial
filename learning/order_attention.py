"""Causal order attention: sub-quadratic value aggregation for K = 2 order heads, for prefill and for decoding.

A K = 2 order head scores query x against key y with the hard-max form of Liu et al. (EMNLP 2023),

    s(x, y) = min(g1(y) - f1(x), g2(y) - f2(x)),

plus a learned sink logit b(x) for key 0 (whose value is v(0)); keys 1..x are visible to query x (causal, self included).
With rank a(x) = f1(x) - f2(x) for queries and r(y) = g1(y) - g2(y) for keys, y is on branch 1 iff r(y) <= a(x), where
s = g1(y) - f1(x); otherwise s = g2(y) - f2(x). So the softmax-weighted value sum over branch 1 is a prefix sum (in rank
order) of exp(g1(y)) v(y), branch 2 a suffix sum of exp(g2(y)) v(y), both scaled by a per-query factor. Sums are kept
as online-softmax states (m, S) with S = sum exp(logit - m) [v, 1]; the trailing channel is the normaliser.

Prefill (all queries known): a position tree. At dyadic level l the keys of every block of 2^l positions are sorted by
rank and chunk-scanned into prefix / suffix states; query x reads one state from each of the <= log N aligned blocks
that tile its past [0, x], and combines them with the sink. O(N d log N) time, differentiable through autograd.

Decoding: ``OrderCache`` keeps the past keys in sorted blocks of sizes 2^i (the logarithmic method). A new token is
inserted as a block of size 1, equal-sized blocks are merged (sort + chunk-scan), and the query reads one state per
block by binary search: O(d log N) amortised per token instead of the O(N d) KV-cache dot product. Position order is
the insertion order, so the causal mask costs nothing at decode time.
"""
import math

import torch

NEG = float("-inf")


def _sortkey(x):
    """Order-preserving float32 -> int64 map into [0, 2^32); -0.0 canonicalised."""
    x = x.float() + 0.0
    b = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(b >= 0x80000000, (~b) & 0xFFFFFFFF, b | 0x80000000)


def combine(m_a, S_a, m_b, S_b):
    """Online-softmax combine of two states; m: (..., 1), S: (..., d). (-inf, 0) is the identity."""
    m = torch.maximum(m_a, m_b)
    # -inf maxima are replaced by finite surrogates before the subtraction so that no NaN is ever formed (a NaN in the
    # unselected branch of torch.where would still poison the backward)
    m_s = torch.where(m == NEG, 0.0, m)
    wa = torch.where(m_a == NEG, 0.0, torch.exp(torch.where(m_a == NEG, 0.0, m_a) - m_s))
    wb = torch.where(m_b == NEG, 0.0, torch.exp(torch.where(m_b == NEG, 0.0, m_b) - m_s))
    return m, S_a * wa + S_b * wb


def chunk_scan(logw, V, chunk=64):
    """Inclusive prefix states along dim -2: (m, S) with S[i] = sum_{j <= i} exp(logw[j] - m[i]) V[j].

    logw: (..., L); V: (..., L, d). Within each chunk the sum is a cumsum stabilised by the chunk max; the carry across
    chunks is a short sequential online-softmax scan over L / chunk states. Returns m (..., L, 1), S (..., L, d).
    """
    *lead, L = logw.shape
    d = V.shape[-1]
    C = min(chunk, L)
    pad = (-L) % C
    if pad:
        logw = torch.nn.functional.pad(logw, (0, pad), value=NEG)
        V = torch.nn.functional.pad(V, (0, 0, 0, pad))
    nc = logw.shape[-1] // C
    lw = logw.view(*lead, nc, C)
    Vc = V.view(*lead, nc, C, d)
    mc = lw.amax(dim=-1, keepdim=True)  # (..., nc, 1)
    # Within a chunk the exponentials are taken in fp64 relative to the chunk max (range e^+-700, so a chunk may span
    # hundreds of nats), then every position is rescaled to its own running max before casting back to fp32: each
    # state (m[i], S[i]) is then a genuine online-softmax state with S of order one, and nothing underflows later.
    M_run = torch.cummax(lw, dim=-1).values.unsqueeze(-1)  # (..., nc, C, 1) running max
    e = torch.exp(lw.double() - torch.where(mc == NEG, 0.0, mc).double())  # exp(-inf) = 0 for excluded keys
    S_in = torch.cumsum(e.unsqueeze(-1) * Vc.double(), dim=-2)  # (..., nc, C, d), relative to mc
    scale = torch.exp(torch.where(mc == NEG, 0.0, mc).double().unsqueeze(-1) - torch.where(M_run == NEG, 0.0, M_run).double())
    S_in = torch.where(M_run == NEG, 0.0, S_in * scale).float()
    # exclusive carry across chunks
    m_run = torch.full_like(mc[..., 0, :], NEG)
    S_run = torch.zeros_like(S_in[..., 0, 0, :])
    m_ex, S_ex = [], []
    for c in range(nc):
        m_ex.append(m_run)
        S_ex.append(S_run)
        m_run, S_run = combine(m_run, S_run, M_run[..., c, -1, :], S_in[..., c, -1, :])  # chunk total, at its max
    m_ex = torch.stack(m_ex, dim=-2).unsqueeze(-2)  # (..., nc, 1, 1)
    S_ex = torch.stack(S_ex, dim=-2).unsqueeze(-2)  # (..., nc, 1, d)
    m_out, S_out = combine(m_ex.expand(*lead, nc, C, 1), S_ex.expand(*lead, nc, C, d), M_run, S_in)
    return m_out.reshape(*lead, L + pad, 1)[..., :L, :], S_out.reshape(*lead, L + pad, d)[..., :L, :]


def chunk_scan_suffix(logw, V, chunk=64):
    """Inclusive suffix states: S[i] = sum_{j >= i} exp(logw[j] - m[i]) V[j]."""
    m, S = chunk_scan(logw.flip(-1), V.flip(-2), chunk)
    return m.flip(-2), S.flip(-2)


def _aug(V):
    return torch.cat([V, torch.ones_like(V[..., :1])], dim=-1)  # trailing channel accumulates the normaliser


def _sort_blocks(rank, size, S):
    """Sort keys inside aligned blocks of ``size`` positions by rank; returns composite keys and the permutation."""
    block = (torch.arange(S, device=rank.device) // size)
    comp = (block << 32) | _sortkey(rank)
    return comp.sort(dim=-1)


def order_attention(f, g, b, V, causal=True, chunk=64):
    """Softmax(order scores) @ V without the N x N matrix.

    Args:
        f: (B, H, N, 2) query realizer values (f1, f2).
        g: (B, H, N, 2) key realizer values (g1, g2); position 0 is the sink key and is scored by ``b`` instead.
        b: (B, H, N) sink logit per query.
        V: (B, H, N, d) values.
        causal: key y is visible to query x iff y <= x. If False every key is visible to every query.
        chunk: chunk size of the scans.

    Returns:
        (B, H, N, d) attention output, identical to ``dense_order_attention``.
    """
    B, H, N, _ = f.shape
    d = V.shape[-1]
    S = 1 << max(N - 1, 0).bit_length()
    padn = S - N
    fp = torch.nn.functional.pad(f.float(), (0, 0, 0, padn))
    gp = torch.nn.functional.pad(g.float(), (0, 0, 0, padn))
    Vp = _aug(torch.nn.functional.pad(V.float(), (0, 0, 0, padn)))
    pos = torch.arange(S, device=f.device)
    is_key = (pos >= 1) & (pos < N)  # the sink (0) and padding never enter the order structure
    rank = torch.where(is_key, gp[..., 0] - gp[..., 1], float("inf"))  # padding sorts last
    rank = torch.where(pos == 0, NEG, rank)  # sink sorts first with zero weight
    lw1 = torch.where(is_key, gp[..., 0], NEG)
    lw2 = torch.where(is_key, gp[..., 1], NEG)
    a = fp[..., 0] - fp[..., 1]
    n = pos + 1  # the past of query x is [0, x]
    levels = range(int(math.log2(S)) + 1) if causal else [int(math.log2(S))]
    m_acc = torch.full((B, H, S, 1), NEG, device=f.device)
    S_acc = torch.zeros((B, H, S, d + 1), device=f.device)
    for l in levels:
        size = 1 << l
        comp, perm = _sort_blocks(rank, size, S)
        g1s, g2s = lw1.gather(-1, perm), lw2.gather(-1, perm)
        Vs = Vp.gather(-2, perm.unsqueeze(-1).expand(-1, -1, -1, d + 1))
        shape = (B, H, S // size, size)
        m1, S1 = chunk_scan(g1s.view(shape), Vs.view(*shape, d + 1), chunk)
        m2, S2 = chunk_scan_suffix(g2s.view(shape), Vs.view(*shape, d + 1), chunk)
        m1, S1, m2, S2 = m1.reshape(B, H, S, 1), S1.reshape(B, H, S, d + 1), m2.reshape(B, H, S, 1), S2.reshape(B, H, S, d + 1)
        if causal:
            has = ((n >> l) & 1) == 1
            start = (n >> (l + 1)) << (l + 1)
        else:
            has = torch.ones(S, dtype=torch.bool, device=f.device)
            start = torch.zeros(S, dtype=torch.int64, device=f.device)
        blk = start // size
        qcomp = (blk << 32) | _sortkey(a)
        r = torch.searchsorted(comp, qcomp, right=True) - start  # keys of the block with rank <= a(x)
        i1 = (start + r - 1).clamp(min=0)
        i2 = (start + r).clamp(max=S - 1)
        ok1 = has & (r > 0)
        ok2 = has & (r < size)
        mm1 = torch.where(ok1.unsqueeze(-1), m1.gather(-2, i1.unsqueeze(-1)) - fp[..., :1], NEG)
        mm2 = torch.where(ok2.unsqueeze(-1), m2.gather(-2, i2.unsqueeze(-1)) - fp[..., 1:2], NEG)
        m_acc, S_acc = combine(m_acc, S_acc, mm1, S1.gather(-2, i1.unsqueeze(-1).expand(-1, -1, -1, d + 1)))
        m_acc, S_acc = combine(m_acc, S_acc, mm2, S2.gather(-2, i2.unsqueeze(-1).expand(-1, -1, -1, d + 1)))
    # sink: logit b(x), value v(0)
    bp = torch.nn.functional.pad(b.float(), (0, padn)).unsqueeze(-1)
    m_acc, S_acc = combine(m_acc, S_acc, bp, Vp[:, :, :1].expand(-1, -1, S, -1))
    out = S_acc[..., :d] / S_acc[..., d:]
    return out[:, :, :N]


def dense_order_attention(f, g, b, V, causal=True):
    """O(N^2) reference: explicit logits, mask, softmax @ V."""
    f, g, V = f.float(), g.float(), V.float()
    s = (g.unsqueeze(2) - f.unsqueeze(3)).amin(dim=-1)  # (B, H, x, y): min_k g_k(y) - f_k(x)
    s = torch.cat([b.float().unsqueeze(-1), s[..., 1:]], dim=-1)
    if causal:
        N = s.shape[-1]
        s = s.masked_fill(~torch.ones(N, N, dtype=torch.bool, device=s.device).tril(), NEG)
    return torch.softmax(s, dim=-1) @ V


def mixture_attention(gates, comps, V, causal=True, chunk=64):
    """omix: query-gated mixture of component order heads; gates (B, H, N, M), comps = [(f, g, b), ...]."""
    w = torch.softmax(gates.float(), dim=-1)
    out = 0
    for i, (f, g, b) in enumerate(comps):
        out = out + w[..., i:i + 1] * order_attention(f, g, b, V, causal, chunk)
    return out


class OrderCache:
    """Order-statistics replacement for a KV cache (one sequence, all heads of one component): logarithmic method.

    Blocks hold keys sorted by rank with prefix (branch 1) and suffix (branch 2) online-softmax states over the
    augmented values. ``prefill`` builds the blocks for a prompt; ``step`` inserts the new token's key (so the token
    attends to itself) and returns its attention output.
    """

    def __init__(self, chunk=64):
        self.chunk = chunk
        self.blocks = []  # list of dicts with rank, g1, g2, V (H, n[, d+1]) and states
        self.v0 = None  # (H, d+1) sink value
        self.n = 0

    def _build(self, rank, g1, g2, V):
        order = rank.argsort(dim=-1)
        rank, g1, g2 = rank.gather(-1, order), g1.gather(-1, order), g2.gather(-1, order)
        V = V.gather(-2, order.unsqueeze(-1).expand(-1, -1, V.shape[-1]))
        m1, S1 = chunk_scan(g1, V, self.chunk)
        m2, S2 = chunk_scan_suffix(g2, V, self.chunk)
        return {"rank": rank, "g1": g1, "g2": g2, "V": V, "m1": m1, "S1": S1, "m2": m2, "S2": S2}

    def _insert_block(self, blk):
        self.blocks.append(blk)
        while len(self.blocks) >= 2 and self.blocks[-1]["rank"].shape[-1] == self.blocks[-2]["rank"].shape[-1]:
            a, c = self.blocks.pop(), self.blocks.pop()
            self.blocks.append(self._build(*(torch.cat([a[k], c[k]], dim=-2 if k == "V" else -1)
                                              for k in ("rank", "g1", "g2", "V"))))

    @torch.no_grad()
    def prefill(self, g, V):
        """g: (H, N, 2) key realizers, V: (H, N, d) values for positions 0..N-1 (0 = sink)."""
        g, V = g.float(), _aug(V.float())
        self.v0 = V[:, 0]
        self.n = g.shape[1]
        rank, g1, g2 = g[:, 1:, 0] - g[:, 1:, 1], g[:, 1:, 0], g[:, 1:, 1]
        V = V[:, 1:]
        start, remaining = 0, rank.shape[1]
        for bit in reversed(range(remaining.bit_length())):  # binary decomposition, largest block first
            size = 1 << bit
            if remaining & size:
                sl = slice(start, start + size)
                self.blocks.append(self._build(rank[:, sl], g1[:, sl], g2[:, sl], V[:, sl]))
                start += size

    @torch.no_grad()
    def step(self, f, g, b, v):
        """Insert token n with key realizer g (H, 2) and value v (H, d); return its output (H, d) for query (f, b)."""
        f, g, v = f.float(), g.float(), _aug(v.float())
        if self.v0 is None:  # first token is the sink
            self.v0, self.n = v, 1
            return self.v0[:, :-1]
        self._insert_block(self._build((g[:, 0] - g[:, 1]).unsqueeze(-1), g[:, :1], g[:, 1:], v.unsqueeze(1)))
        self.n += 1
        return self.query(f, b)

    @torch.no_grad()
    def query(self, f, b):
        H = f.shape[0]
        d1 = self.v0.shape[-1]
        a = (f[:, 0] - f[:, 1]).unsqueeze(-1)  # (H, 1)
        m_acc = torch.full((H, 1), NEG, device=f.device)
        S_acc = torch.zeros((H, d1), device=f.device)
        for blk in self.blocks:
            n = blk["rank"].shape[-1]
            r = torch.searchsorted(blk["rank"].contiguous(), a.contiguous(), right=True)  # (H, 1)
            i1, i2 = (r - 1).clamp(min=0), r.clamp(max=n - 1)
            ok1, ok2 = r > 0, r < n
            mm1 = torch.where(ok1, blk["m1"].gather(-2, i1.unsqueeze(-1)).squeeze(-1) - f[:, :1], NEG)
            mm2 = torch.where(ok2, blk["m2"].gather(-2, i2.unsqueeze(-1)).squeeze(-1) - f[:, 1:2], NEG)
            m_acc, S_acc = combine(m_acc, S_acc, mm1, blk["S1"].gather(-2, i1.unsqueeze(-1).expand(-1, -1, d1)).squeeze(-2))
            m_acc, S_acc = combine(m_acc, S_acc, mm2, blk["S2"].gather(-2, i2.unsqueeze(-1).expand(-1, -1, d1)).squeeze(-2))
        m_acc, S_acc = combine(m_acc, S_acc, b.float().unsqueeze(-1), self.v0)
        return S_acc[:, :-1] / S_acc[:, -1:]
