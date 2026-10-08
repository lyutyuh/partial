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

Decoding: ``OrderCache`` keeps the prefilled keys in one static rank-sorted block and the keys appended while decoding
in a tail of sorted blocks of sizes 2^l (the logarithmic method), all in one preallocated slot buffer. A new token
carries into the lowest empty tail level (sort + chunk-scan of 2^j keys), a full tail is folded into the static block,
and a query reads one prefix and one suffix state per segment with a single batched binary search: O(d log N) per
token in a fixed number of kernel launches (or one CUDA-graph replay), instead of the O(N d) KV-cache dot product.
Position order is the insertion order, so the causal mask costs nothing at decode time. ``OrderCacheSimple`` is the
earlier block-list implementation, kept for comparison.
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


class OrderCacheSimple:
    """Order-statistics replacement for a KV cache (one sequence, all heads of one component): logarithmic method.

    Blocks hold keys sorted by rank with prefix (branch 1) and suffix (branch 2) online-softmax states over the
    augmented values. ``prefill`` builds the blocks for a prompt; ``step`` inserts the new token's key (so the token
    attends to itself) and returns its attention output.

    Reference implementation kept for comparison with ``OrderCache``: one Python-level block list, a query costs a few
    kernel launches per block, and a prefill of 2^k tokens leaves one block of every size, so the first decode step
    cascades into a rebuild of the whole cache.
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


_EMPTY = 0xFFFFFFFF  # low word of an empty slot's composite key: above the _sortkey of every non-NaN float


def _chunk_states(lw, V):
    """Inclusive prefix states inside each chunk along dim -1 of lw (..., C): ``chunk_scan``'s in-chunk arithmetic.

    The exponentials are taken in fp64 relative to the chunk max and every position is rescaled to its own running max
    before the cast back to fp32. Returns m (..., C, 1), S (..., C, d) for V (..., C, d).
    """
    M_run = torch.cummax(lw, dim=-1).values.unsqueeze(-1)  # running max
    mc = M_run[..., -1:, :]  # chunk max
    mc0 = torch.where(mc == NEG, 0.0, mc).double()
    S = torch.cumsum(torch.exp(lw.double().unsqueeze(-1) - mc0) * V.double(), dim=-2)
    S = torch.where(M_run == NEG, 0.0, S * torch.exp(mc0 - torch.where(M_run == NEG, 0.0, M_run).double()))
    return M_run, S.float()


def _scan_tree(logw, V, chunk=64):
    """Inclusive prefix states, equal to ``chunk_scan`` but with a log-depth carry across chunks.

    Every chunk is scanned on its own (``_chunk_states``, same fp64 arithmetic as ``chunk_scan``); the chunk totals are
    then scanned by recursive doubling (log2(L / chunk) vectorised combines instead of a Python loop over the chunks),
    and each chunk is combined with the total of all chunks before it. logw: (..., L); V: (..., L, d).
    """
    *lead, L = logw.shape
    d = V.shape[-1]
    C = min(chunk, L)
    pad = (-L) % C
    if pad:
        logw = torch.nn.functional.pad(logw, (0, pad), value=NEG)
        V = torch.nn.functional.pad(V, (0, 0, 0, pad))
    nc = logw.shape[-1] // C
    m, S = _chunk_states(logw.reshape(*lead, nc, C), V.reshape(*lead, nc, C, d))  # (..., nc, C, 1), (..., nc, C, d)
    if nc > 1:
        mt, St = m[..., -1, :], S[..., -1, :]  # chunk totals, (..., nc, 1) and (..., nc, d)
        k = 1
        while k < nc:  # inclusive scan of the totals: after the pass with stride k, state i covers chunks (i-2k, i]
            mk, Sk = combine(mt[..., :-k, :], St[..., :-k, :], mt[..., k:, :], St[..., k:, :])
            mt, St = torch.cat([mt[..., :k, :], mk], dim=-2), torch.cat([St[..., :k, :], Sk], dim=-2)
            k *= 2
        m_ex = mt[..., :-1, None, :].expand(*lead, nc - 1, C, 1)  # carry into chunk c: total of chunks < c
        S_ex = St[..., :-1, None, :].expand(*lead, nc - 1, C, d)
        mc, Sc = combine(m_ex, S_ex, m[..., 1:, :, :], S[..., 1:, :, :])
        m, S = torch.cat([m[..., :1, :, :], mc], dim=-3), torch.cat([S[..., :1, :, :], Sc], dim=-3)
    return m.reshape(*lead, nc * C, 1)[..., :L, :], S.reshape(*lead, nc * C, d)[..., :L, :]


class OrderCache:
    """Decode-time replacement of a KV cache for one K = 2 order head group (one sequence, H heads).

    Layout: ONE preallocated slot buffer per head, cut into segments laid out in id order. Segment l < L is tail level
    l (capacity 2^l: the logarithmic method over the keys appended while decoding); segment L is the static block
    (capacity Ns: every prefilled key and every folded tail). Each segment starts with a guard slot, and a trailing
    guard closes the buffer. Inside a segment the keys sit sorted by rank, and every slot holds the raw key
    R = [v, 1, g1, g2] plus the packed inclusive prefix state P1 = [S1, m1] of exp(g1) [v, 1] and suffix state
    P2 = [S2, m2] of exp(g2) [v, 1]. The composite int64 key of a slot is (segment << 32 | _sortkey(rank)); an empty
    slot has low word 0xFFFFFFFF, a guard low word 0 (the trailing guard is (L + 1) << 32), so the keys are sorted along
    the whole buffer.

    A query therefore runs ONE batched searchsorted of its key (segment << 32 | _sortkey(a)) for all L + 1 segments:
    the slot before the hit is the segment's last branch-1 key (or its guard) and the hit is its first branch-2 key (or
    an empty slot or the next guard). Guards and empty slots hold the identity state (m = -inf), so two gathers and one
    softmax over the 2 (L + 1) states and the sink give the output with no masking: a fixed number of kernel launches
    for any N, optionally one CUDA-graph replay.

    Insertion is a binary counter over the tail: the new key carries into the lowest empty level j, rebuilt (sort +
    chunk-scan) from the full levels 0..j-1 and the new key, so a step rebuilds 2^j keys with probability 2^-(j+1).
    Prefill writes one static block, so no carry cascade follows it. When the tail holds C = min(tail_capacity,
    2^L - 1) keys, the next step folds the tail and the new key into the static block (one rebuild of the cache, every
    C steps) and reallocates the buffer. L adapts to the static size: 2^L - 1 is the largest such value <= Ns / 4,
    clipped to [min_tail_capacity, tail_capacity], so the tail costs <= ~25 % memory on long contexts and folding
    costs O(1) amortised key rebuilds per step.

    Memory (``nbytes``): 12 d + 36 bytes per slot and head (fp32 raw key and two fp32 states plus the int64 key; 1572
    B at d = 128, 3.1x a bf16 K + V pair of the same head), times ~1.25 for the preallocated tail. One cache serves
    ONE K = 2 component per head and cannot share values across GQA query heads (each head sorts by its own rank), so
    against a GQA KV cache with G query heads per KV head the ratio grows by G, and by M for an M-component mixture.
    Not in ``nbytes``: the CUDA-graph pool (all graphs share one), and the transient peak of a prefill or fold (copies
    of the raw keys plus fp64 scan temporaries bounded by ``budget``), about 2x the steady state on long contexts.

    Args:
        chunk: chunk size of the scans that build segment states.
        tail_capacity: upper bound on the keys the tail holds before a fold (levels 0..L-1 with
            L <= tail_capacity.bit_length(); 2^k - 1 uses every slot).
        min_tail_capacity: lower bound on the tail capacity for short contexts.
        cuda_graph: replay the query, and every step whose merge level is <= ``graph_max_level``, from CUDA graphs
            captured lazily (CUDA only). A fold reallocates the buffer, so the graphs are recaptured after it.
        graph_max_level: deepest merge level captured in a step graph (2^level keys); deeper merges (rare) run eagerly.
    """

    def __init__(self, chunk=64, tail_capacity=(1 << 16) - 1, min_tail_capacity=1023, cuda_graph=False,
                 graph_max_level=8):
        self.chunk = chunk
        self.tail_capacity = tail_capacity
        self.min_tail_capacity = min_tail_capacity
        self.cuda_graph = cuda_graph
        self.graph_max_level = graph_max_level
        self.budget = 1 << 24  # elements (2 x heads x keys x (d+1)) scanned at once: bounds the fp64 temporaries
        self.v0 = None  # (H, d+1) sink value
        self.n = 0  # tokens seen, sink included
        self.ns = 0  # keys in the static block
        self.count = 0  # keys in the tail; bit l set iff tail level l is full
        self.rebuilt = 0  # keys sorted + scanned so far (prefill, merges, folds): the work counter of the tests
        self.folds = 0
        self.captures = 0
        self._graphs = {}

    # ---------------------------------------------------------------------------------------------------- layout
    def _levels(self, ns):
        """Tail levels L for a static block of ns keys: 2^L - 1 ~ ns / 4, within the capacity bounds."""
        lmax = max(1, self.tail_capacity.bit_length())
        lmin = min(self.min_tail_capacity.bit_length(), lmax)
        return min(lmax, max(lmin, (ns // 4 + 1).bit_length() - 1))

    @staticmethod
    def _slots(level):
        """Data slots [lo, hi) of tail level ``level``; its guard is slot lo - 1."""
        return (1 << level) + level, (2 << level) + level

    def _alloc(self, H, d, ns, device):
        """(Re)allocate the buffer for a static block of ``ns`` keys and an empty tail; drops the captured graphs."""
        L = self._levels(ns)
        self.L, self.C = L, min(self.tail_capacity, (1 << L) - 1)
        self.base = (1 << L) + L  # first static data slot (its guard is base - 1)
        size = self.base + ns + 1
        lens = torch.tensor([(1 << level) + 1 for level in range(L)] + [ns + 1, 1])  # guard + data per segment
        seg = torch.repeat_interleave(torch.arange(L + 2), lens)
        keys = (seg << 32) | _EMPTY
        guards = torch.cumsum(lens, 0) - lens
        keys[guards] = seg[guards] << 32
        self._empty = keys.unsqueeze(0).to(device)  # (1, size) composite keys of the empty buffer
        self.comp = self._empty.expand(H, -1).clone()
        self.R = torch.zeros(H, size, d + 3, device=device)
        self.P1 = torch.zeros(H, size, d + 2, device=device)
        self.P2 = torch.zeros(H, size, d + 2, device=device)
        self.P1[..., -1] = NEG  # every guard and empty slot is the identity state
        self.P2[..., -1] = NEG
        self.segkey = (torch.arange(L + 1, device=device) << 32).unsqueeze(0)  # (1, L + 1)
        # data slots of levels 0..j-1, read by a carry into level j
        self._below = [torch.cat([torch.arange(*self._slots(level)) for level in range(j)]).to(device) if j else None
                       for j in range(L)]
        self.ns, self.count = 0, 0
        self._graphs = {}
        self._pool = None  # the graphs of one buffer share a private memory pool; a fresh pool after a reallocation
        self._f_in = torch.zeros(H, 2, device=device)  # static inputs of the CUDA graphs
        self._g_in = torch.zeros(H, 2, device=device)
        self._b_in = torch.zeros(H, device=device)
        self._v_in = torch.zeros(H, d, device=device)
        self._one = torch.ones(H, 1, device=device)

    def nbytes(self):
        """Bytes of the slot buffers: static block and preallocated tail (no graph pool, no transients)."""
        bufs = (self.comp, self.R, self.P1, self.P2, self.v0)
        return sum(t.numel() * t.element_size() for t in bufs)

    def segments(self):
        """Host-side occupancy [(segment, keys)]: the tail levels, then the static block."""
        tail = [(level, (1 << level) if (self.count >> level) & 1 else 0) for level in range(self.L)]
        return tail + [(self.L, self.ns)]

    def _write(self, off, seg, R):
        """Sort n raw keys by rank and write them with their prefix / suffix states to the slots [off, off + n).

        R: (H, n, d+3) raw keys [v, 1, g1, g2]; seg: the segment id of the slots. Head groups bound the temporaries.
        """
        H, n, e = R.shape
        key = _sortkey(R[..., -2] - R[..., -1])
        if n > 1:
            key, perm = key.sort(dim=-1)
            R = R.gather(1, perm.unsqueeze(-1).expand(-1, -1, e))
        sl = slice(off, off + n)
        self.comp[:, sl] = key | (seg << 32)
        self.R[:, sl] = R
        if n == 1:  # a single key is its own prefix and suffix state: [v, 1, g1] and [v, 1, g2]
            self.P1[:, sl] = R[..., :-1]
            self.P2[:, sl, :-1], self.P2[:, sl, -1] = R[..., :-2], R[..., -1]
            return
        hg = max(1, self.budget // (2 * n * (e - 2)))
        for h in range(0, H, hg):
            hs = slice(h, h + hg)
            Va = R[hs, :, :-2]  # [v, 1]
            # prefix scan of exp(g1) and suffix scan (= prefix scan of the reversal) of exp(g2) in one call
            m, S = _scan_tree(torch.stack([R[hs, :, -2], R[hs, :, -1].flip(-1)]), torch.stack([Va, Va.flip(-2)]),
                              self.chunk)
            self.P1[hs, sl, :-1], self.P1[hs, sl, -1] = S[0], m[0, ..., 0]
            self.P2[hs, sl, :-1], self.P2[hs, sl, -1] = S[1].flip(-2), m[1, ..., 0].flip(-1)

    # ---------------------------------------------------------------------------------------------------- updates
    @torch.no_grad()
    def prefill(self, g, V):
        """g: (H, N, 2) key realizers, V: (H, N, d) values for positions 0..N-1 (0 = sink); resets the cache.

        All N - 1 keys go into one static block: the tail starts empty, so the first decode steps rebuild O(1) keys.
        """
        g, V = g.float(), V.float()
        H, N, d = V.shape
        self.v0 = _aug(V[:, 0])
        self._alloc(H, d, N - 1, V.device)
        self.n = N
        if N > 1:
            self._write(self.base, self.L, torch.cat([_aug(V[:, 1:]), g[:, 1:]], dim=-1))
            self.ns = N - 1
            self.rebuilt += N - 1

    def _merge(self, j, g, v):
        """Binary-counter carry: rebuild tail level j from the full levels 0..j-1 and the new key g (H, 2), v (H, d).

        The raw keys of the merged levels stay in place (only their composite keys and suffix maxima are reset), so
        running this twice from the same state is harmless, which the CUDA-graph warm-up relies on.
        """
        R = torch.cat([v, self._one, g], dim=-1).unsqueeze(1)
        if j:
            R = torch.cat([self.R.index_select(1, self._below[j]), R], dim=1)
        self._write(self._slots(j)[0], j, R)
        if j:
            end = self._slots(j)[0] - 1  # levels 0..j-1 and their guards: slots [0, end)
            self.comp[:, :end] = self._empty[:, :end]
            self.P2[:, :end, -1] = NEG  # emptied slots read as identity suffix states

    def _fold(self, g, v):
        """Fold the tail and the new key into the static block: one rebuild of the cache, then an empty tail."""
        sls = [slice(self.base, self.base + self.ns)]
        sls += [slice(*self._slots(level)) for level in range(self.L) if (self.count >> level) & 1]
        R = torch.cat([self.R[:, sl] for sl in sls] + [torch.cat([v, self._one, g], dim=-1).unsqueeze(1)], dim=1)
        H, n, e = R.shape
        self.comp = self.R = self.P1 = self.P2 = self._graphs = None  # free the old buffer before reallocating
        self._alloc(H, e - 3, n, R.device)
        self._write(self.base, self.L, R)
        self.ns = n
        self.rebuilt += n
        self.folds += 1

    def _replay(self, key, fn):
        """Run ``fn`` from a CUDA graph, capturing it on first use (after one idempotent warm-up run).

        All graphs of the current buffer are captured into ONE private memory pool (one pool per graph held ~13 MiB
        each). Sharing is safe because replays are serialised on one stream and every caller clones the static output
        right after the replay, before another graph can reuse that memory for its temporaries.
        """
        if key not in self._graphs:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                fn()
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                out = fn()
            self._pool = graph.pool()
            self._graphs[key] = (graph, out)
            self.captures += 1
        graph, out = self._graphs[key]
        graph.replay()
        return out

    def _graphs_on(self, x):
        return self.cuda_graph and x.is_cuda

    @torch.no_grad()
    def step(self, f, g, b, v):
        """Insert token n with key realizer g (H, 2) and value v (H, d); return its output (H, d) for query (f, b)."""
        f, g, b, v = f.float(), g.float(), b.float(), v.float()
        if self.v0 is None:  # first token is the sink
            self.prefill(g.unsqueeze(1), v.unsqueeze(1))
            return v.clone()
        self.n += 1
        if self.count == self.C:
            self._fold(g, v)
            return self.query(f, b)
        j = ((self.count + 1) & ~self.count).bit_length() - 1  # lowest empty tail level
        self.count += 1
        self.rebuilt += 1 << j
        if self._graphs_on(f) and j <= self.graph_max_level:
            torch._foreach_copy_([self._g_in, self._v_in, self._f_in, self._b_in], [g, v, f, b])  # one launch

            def run():
                self._merge(j, self._g_in, self._v_in)
                return self._query(self._f_in, self._b_in)

            return self._replay(("step", j), run).clone()
        self._merge(j, g, v)
        return self.query(f, b)

    def _query(self, f, b):
        a = f[:, 0] - f[:, 1]
        p = torch.searchsorted(self.comp, self.segkey | _sortkey(a).unsqueeze(-1), right=True)  # (H, L + 1)
        e = self.P1.shape[-1]
        P1 = self.P1.gather(1, (p - 1).unsqueeze(-1).expand(-1, -1, e))  # last branch-1 key of each segment, or guard
        P2 = self.P2.gather(1, p.unsqueeze(-1).expand(-1, -1, e))  # first branch-2 key, or an empty slot / next guard
        logit = torch.cat([P1[..., -1] - f[:, :1], P2[..., -1] - f[:, 1:], b.unsqueeze(-1)], dim=-1)
        w = torch.exp(logit - logit.amax(dim=-1, keepdim=True))  # identity states get weight exactly 0
        S = torch.cat([P1[..., :-1], P2[..., :-1], self.v0.unsqueeze(1)], dim=1)
        out = (w.unsqueeze(-1) * S).sum(dim=1)
        return out[:, :-1] / out[:, -1:]

    @torch.no_grad()
    def query(self, f, b):
        """Attention output (H, d) of query realizers f (H, 2) and sink logit b (H,) over the cached keys."""
        f, b = f.float(), b.float()
        if self._graphs_on(f):
            torch._foreach_copy_([self._f_in, self._b_in], [f, b])
            return self._replay(("query",), lambda: self._query(self._f_in, self._b_in)).clone()
        return self._query(f, b)
