"""Causal order attention: sub-quadratic value aggregation for K = 2 order heads, for prefill and for decoding.

A K = 2 order head scores query x against key y with the hard-max form of Liu et al. (EMNLP 2023),

    s(x, y) = min(g1(y) - f1(x), g2(y) - f2(x)),

plus a learned sink logit b(x) for key 0 (whose value is v(0)); keys 1..x are visible to query x (causal, self included).
With rank a(x) = f1(x) - f2(x) for queries and r(y) = g1(y) - g2(y) for keys, y is on branch 1 iff r(y) <= a(x), where
s = g1(y) - f1(x); otherwise s = g2(y) - f2(x). So the softmax-weighted value sum over branch 1 is a prefix sum (in rank
order) of exp(g1(y)) v(y), branch 2 a suffix sum of exp(g2(y)) v(y), both scaled by a per-query factor. Sums are kept
as online-softmax states (m, S) with S = sum exp(logit - m) [v, 1]; the trailing channel is the normaliser.

Layout: f, g (batch, num_heads, seq_len, 2); b (batch, num_heads, seq_len); values (batch, num_heads, seq_len, head_dim),
the internal layout of the transformers attention modules. The Triton kernels of ``order_attention_triton`` implement
the same semantics; this module is their verified reference and the only path with higher-order gradients.

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
import logging
import math
from typing import Callable, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

NEG = float("-inf")

Component = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
"""One K = 2 order head: (f, g, b) realizers and sink logits."""


def _sortkey(x: torch.Tensor) -> torch.Tensor:
    """Order-preserving float32 -> int64 map into [0, 2^32); -0.0 canonicalised."""
    x = x.float() + 0.0
    bits = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where(bits >= 0x80000000, (~bits) & 0xFFFFFFFF, bits | 0x80000000)


def online_softmax_combine(
    m_a: torch.Tensor, S_a: torch.Tensor, m_b: torch.Tensor, S_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online-softmax combine of two states; m: (..., 1), S: (..., d). (-inf, 0) is the identity."""
    m = torch.maximum(m_a, m_b)
    # -inf maxima are replaced by finite surrogates before the subtraction so that no NaN is ever formed (a NaN in the
    # unselected branch of torch.where would still poison the backward)
    m_s = torch.where(m == NEG, 0.0, m)
    w_a = torch.where(m_a == NEG, 0.0, torch.exp(torch.where(m_a == NEG, 0.0, m_a) - m_s))
    w_b = torch.where(m_b == NEG, 0.0, torch.exp(torch.where(m_b == NEG, 0.0, m_b) - m_s))
    return m, S_a * w_a + S_b * w_b


def chunk_scan(logw: torch.Tensor, value: torch.Tensor, chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive prefix states along dim -2: (m, S) with S[i] = sum_{j <= i} exp(logw[j] - m[i]) value[j].

    Within each chunk the sum is a cumsum stabilised by the chunk max; the carry across chunks is a short sequential
    online-softmax scan over L / chunk states.

    Args:
        logw: (..., L) log-weights; -inf excludes a position.
        value: (..., L, d) values.
        chunk: chunk length of the scan.

    Returns:
        m (..., L, 1) running maxima and S (..., L, d) the states relative to them.
    """
    *lead, length = logw.shape
    d = value.shape[-1]
    chunk_len = min(chunk, length)
    pad = (-length) % chunk_len
    if pad:
        logw = torch.nn.functional.pad(logw, (0, pad), value=NEG)
        value = torch.nn.functional.pad(value, (0, 0, 0, pad))
    nchunks = logw.shape[-1] // chunk_len
    logw_c = logw.view(*lead, nchunks, chunk_len)
    value_c = value.view(*lead, nchunks, chunk_len, d)
    chunk_max = logw_c.amax(dim=-1, keepdim=True)  # (..., nchunks, 1)
    # Within a chunk the exponentials are taken in fp64 relative to the chunk max (range e^+-700, so a chunk may span
    # hundreds of nats), then every position is rescaled to its own running max before casting back to fp32: each
    # state (m[i], S[i]) is then a genuine online-softmax state with S of order one, and nothing underflows later.
    running_max = torch.cummax(logw_c, dim=-1).values.unsqueeze(-1)  # (..., nchunks, chunk_len, 1)
    weights = torch.exp(logw_c.double() - torch.where(chunk_max == NEG, 0.0, chunk_max).double())  # exp(-inf) = 0
    S_chunk = torch.cumsum(weights.unsqueeze(-1) * value_c.double(), dim=-2)  # relative to chunk_max
    scale = torch.exp(
        torch.where(chunk_max == NEG, 0.0, chunk_max).double().unsqueeze(-1)
        - torch.where(running_max == NEG, 0.0, running_max).double()
    )
    S_chunk = torch.where(running_max == NEG, 0.0, S_chunk * scale).float()
    # exclusive carry across chunks
    carry_m = torch.full_like(chunk_max[..., 0, :], NEG)
    carry_S = torch.zeros_like(S_chunk[..., 0, 0, :])
    m_excl, S_excl = [], []
    for c in range(nchunks):
        m_excl.append(carry_m)
        S_excl.append(carry_S)
        carry_m, carry_S = online_softmax_combine(
            carry_m, carry_S, running_max[..., c, -1, :], S_chunk[..., c, -1, :]
        )  # chunk total, at its max
    m_excl = torch.stack(m_excl, dim=-2).unsqueeze(-2)  # (..., nchunks, 1, 1)
    S_excl = torch.stack(S_excl, dim=-2).unsqueeze(-2)  # (..., nchunks, 1, d)
    m_out, S_out = online_softmax_combine(
        m_excl.expand(*lead, nchunks, chunk_len, 1), S_excl.expand(*lead, nchunks, chunk_len, d), running_max, S_chunk
    )
    m_out = m_out.reshape(*lead, length + pad, 1)[..., :length, :]
    S_out = S_out.reshape(*lead, length + pad, d)[..., :length, :]
    return m_out, S_out


def chunk_scan_suffix(logw: torch.Tensor, value: torch.Tensor, chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive suffix states: S[i] = sum_{j >= i} exp(logw[j] - m[i]) value[j]."""
    m, S = chunk_scan(logw.flip(-1), value.flip(-2), chunk)
    return m.flip(-2), S.flip(-2)


def _append_ones(value: torch.Tensor) -> torch.Tensor:
    """[v, 1]: the trailing channel accumulates the normaliser."""
    return torch.cat([value, torch.ones_like(value[..., :1])], dim=-1)


def _sort_within_blocks(rank: torch.Tensor, block_size: int, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort keys inside aligned blocks of ``block_size`` positions by rank; returns (composite keys, permutation)."""
    block = torch.arange(seqlen, device=rank.device) // block_size
    composite = (block << 32) | _sortkey(rank)
    return composite.sort(dim=-1)


def order_attention(
    f: torch.Tensor, g: torch.Tensor, b: torch.Tensor, value: torch.Tensor, causal: bool = True, chunk: int = 64
) -> torch.Tensor:
    """Softmax(order scores) @ value without the N x N matrix (a position tree of chunk scans).

    Args:
        f: (batch, num_heads, seq_len, 2) query realizer values (f1, f2).
        g: (batch, num_heads, seq_len, 2) key realizer values (g1, g2); position 0 is the sink key and is scored by
            ``b`` instead.
        b: (batch, num_heads, seq_len) sink logit per query.
        value: (batch, num_heads, seq_len, head_dim) values.
        causal: key y is visible to query x iff y <= x. If False every key is visible to every query.
        chunk: chunk size of the scans.

    Returns:
        (batch, num_heads, seq_len, head_dim) attention output, identical to ``dense_order_attention``.
    """
    batch, num_heads, seq_len, _ = f.shape
    head_dim = value.shape[-1]
    seqlen_padded = 1 << max(seq_len - 1, 0).bit_length()
    pad = seqlen_padded - seq_len
    f_pad = torch.nn.functional.pad(f.float(), (0, 0, 0, pad))
    g_pad = torch.nn.functional.pad(g.float(), (0, 0, 0, pad))
    v_pad = _append_ones(torch.nn.functional.pad(value.float(), (0, 0, 0, pad)))
    pos = torch.arange(seqlen_padded, device=f.device)
    is_key = (pos >= 1) & (pos < seq_len)  # the sink (0) and padding never enter the order structure
    rank = torch.where(is_key, g_pad[..., 0] - g_pad[..., 1], float("inf"))  # padding sorts last
    rank = torch.where(pos == 0, NEG, rank)  # sink sorts first with zero weight
    logw1 = torch.where(is_key, g_pad[..., 0], NEG)
    logw2 = torch.where(is_key, g_pad[..., 1], NEG)
    qrank = f_pad[..., 0] - f_pad[..., 1]
    n = pos + 1  # the past of query x is [0, x]
    levels = range(int(math.log2(seqlen_padded)) + 1) if causal else [int(math.log2(seqlen_padded))]
    m_acc = torch.full((batch, num_heads, seqlen_padded, 1), NEG, device=f.device)
    S_acc = torch.zeros((batch, num_heads, seqlen_padded, head_dim + 1), device=f.device)
    for level in levels:
        block_size = 1 << level
        composite, perm = _sort_within_blocks(rank, block_size, seqlen_padded)
        logw1_sorted, logw2_sorted = logw1.gather(-1, perm), logw2.gather(-1, perm)
        v_sorted = v_pad.gather(-2, perm.unsqueeze(-1).expand(-1, -1, -1, head_dim + 1))
        shape = (batch, num_heads, seqlen_padded // block_size, block_size)
        m1, S1 = chunk_scan(logw1_sorted.view(shape), v_sorted.view(*shape, head_dim + 1), chunk)
        m2, S2 = chunk_scan_suffix(logw2_sorted.view(shape), v_sorted.view(*shape, head_dim + 1), chunk)
        m1 = m1.reshape(batch, num_heads, seqlen_padded, 1)
        S1 = S1.reshape(batch, num_heads, seqlen_padded, head_dim + 1)
        m2 = m2.reshape(batch, num_heads, seqlen_padded, 1)
        S2 = S2.reshape(batch, num_heads, seqlen_padded, head_dim + 1)
        if causal:
            has = ((n >> level) & 1) == 1
            start = (n >> (level + 1)) << (level + 1)
        else:
            has = torch.ones(seqlen_padded, dtype=torch.bool, device=f.device)
            start = torch.zeros(seqlen_padded, dtype=torch.int64, device=f.device)
        block = start // block_size
        qkey = (block << 32) | _sortkey(qrank)
        split = torch.searchsorted(composite, qkey, right=True) - start  # keys of the block with rank <= a(x)
        row1 = (start + split - 1).clamp(min=0)
        row2 = (start + split).clamp(max=seqlen_padded - 1)
        has1 = has & (split > 0)
        has2 = has & (split < block_size)
        logit1 = torch.where(has1.unsqueeze(-1), m1.gather(-2, row1.unsqueeze(-1)) - f_pad[..., :1], NEG)
        logit2 = torch.where(has2.unsqueeze(-1), m2.gather(-2, row2.unsqueeze(-1)) - f_pad[..., 1:2], NEG)
        m_acc, S_acc = online_softmax_combine(
            m_acc, S_acc, logit1, S1.gather(-2, row1.unsqueeze(-1).expand(-1, -1, -1, head_dim + 1))
        )
        m_acc, S_acc = online_softmax_combine(
            m_acc, S_acc, logit2, S2.gather(-2, row2.unsqueeze(-1).expand(-1, -1, -1, head_dim + 1))
        )
    # sink: logit b(x), value v(0)
    b_pad = torch.nn.functional.pad(b.float(), (0, pad)).unsqueeze(-1)
    m_acc, S_acc = online_softmax_combine(m_acc, S_acc, b_pad, v_pad[:, :, :1].expand(-1, -1, seqlen_padded, -1))
    out = S_acc[..., :head_dim] / S_acc[..., head_dim:]
    return out[:, :, :seq_len]


def eager_order_attention_forward(
    f: torch.Tensor, g: torch.Tensor, b: torch.Tensor, value: torch.Tensor, causal: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """O(N^2) reference in the style of ``eager_attention_forward``: explicit logits, mask, softmax @ value.

    Args:
        f, g, b, value: as in ``order_attention``.
        causal: mask keys y > x.

    Returns:
        attn_output (batch, num_heads, seq_len, head_dim) and attn_weights (batch, num_heads, seq_len, seq_len).
    """
    f, g, value = f.float(), g.float(), value.float()
    scores = (g.unsqueeze(2) - f.unsqueeze(3)).amin(dim=-1)  # (batch, num_heads, x, y): min_k g_k(y) - f_k(x)
    scores = torch.cat([b.float().unsqueeze(-1), scores[..., 1:]], dim=-1)
    if causal:
        seq_len = scores.shape[-1]
        scores = scores.masked_fill(~torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device).tril(), NEG)
    attn_weights = torch.softmax(scores, dim=-1)
    attn_output = attn_weights @ value
    return attn_output, attn_weights


def dense_order_attention(
    f: torch.Tensor, g: torch.Tensor, b: torch.Tensor, value: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """O(N^2) reference: the attention output of ``eager_order_attention_forward``."""
    return eager_order_attention_forward(f, g, b, value, causal)[0]


def mixture_attention(
    gates: torch.Tensor, components: Sequence[Component], value: torch.Tensor, causal: bool = True, chunk: int = 64
) -> torch.Tensor:
    """omix: query-gated mixture of component order heads.

    Args:
        gates: (batch, num_heads, seq_len, M) mixture logits.
        components: M tuples (f, g, b) as in ``order_attention``.
        value: (batch, num_heads, seq_len, head_dim) values shared by the components.
        causal, chunk: see ``order_attention``.

    Returns:
        (batch, num_heads, seq_len, head_dim) sum_m softmax(gates)_m order_attention(*components[m], value).
    """
    weights = torch.softmax(gates.float(), dim=-1)
    out = 0
    for i, (f, g, b) in enumerate(components):
        out = out + weights[..., i : i + 1] * order_attention(f, g, b, value, causal, chunk)
    return out


class OrderCacheSimple:
    """Order-statistics replacement for a KV cache (one sequence, all heads of one component): logarithmic method.

    Blocks hold keys sorted by rank with prefix (branch 1) and suffix (branch 2) online-softmax states over the
    augmented values. ``prefill`` builds the blocks for a prompt; ``step`` inserts the new token's key (so the token
    attends to itself) and returns its attention output.

    Reference implementation kept for comparison with ``OrderCache``: one Python-level block list, a query costs a few
    kernel launches per block, and a prefill of 2^k tokens leaves one block of every size, so the first decode step
    cascades into a rebuild of the whole cache.

    Args:
        chunk: chunk size of the scans that build block states.
    """

    def __init__(self, chunk: int = 64):
        self.chunk = chunk
        self.blocks: list[dict[str, torch.Tensor]] = []  # rank, g1, g2, V (H, n[, d+1]) and the states of each block
        self.v0: Optional[torch.Tensor] = None  # (H, d+1) sink value
        self.n = 0

    def _build_block(
        self, rank: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor, value: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        order = rank.argsort(dim=-1)
        rank, g1, g2 = rank.gather(-1, order), g1.gather(-1, order), g2.gather(-1, order)
        value = value.gather(-2, order.unsqueeze(-1).expand(-1, -1, value.shape[-1]))
        m1, S1 = chunk_scan(g1, value, self.chunk)
        m2, S2 = chunk_scan_suffix(g2, value, self.chunk)
        return {"rank": rank, "g1": g1, "g2": g2, "V": value, "m1": m1, "S1": S1, "m2": m2, "S2": S2}

    def _insert_block(self, block: dict[str, torch.Tensor]) -> None:
        self.blocks.append(block)
        while len(self.blocks) >= 2 and self.blocks[-1]["rank"].shape[-1] == self.blocks[-2]["rank"].shape[-1]:
            a, c = self.blocks.pop(), self.blocks.pop()
            merged = (torch.cat([a[k], c[k]], dim=-2 if k == "V" else -1) for k in ("rank", "g1", "g2", "V"))
            self.blocks.append(self._build_block(*merged))

    @torch.no_grad()
    def prefill(self, g: torch.Tensor, value: torch.Tensor) -> None:
        """Builds the blocks of a prompt.

        Args:
            g: (H, N, 2) key realizers of positions 0..N-1 (0 = sink).
            value: (H, N, d) values.
        """
        g, value = g.float(), _append_ones(value.float())
        self.v0 = value[:, 0]
        self.n = g.shape[1]
        rank, g1, g2 = g[:, 1:, 0] - g[:, 1:, 1], g[:, 1:, 0], g[:, 1:, 1]
        value = value[:, 1:]
        start, remaining = 0, rank.shape[1]
        for bit in reversed(range(remaining.bit_length())):  # binary decomposition, largest block first
            size = 1 << bit
            if remaining & size:
                rows = slice(start, start + size)
                self.blocks.append(self._build_block(rank[:, rows], g1[:, rows], g2[:, rows], value[:, rows]))
                start += size

    @torch.no_grad()
    def step(self, f: torch.Tensor, g: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Inserts token n and returns its attention output.

        Args:
            f: (H, 2) query realizers of the new token.
            g: (H, 2) its key realizers.
            b: (H,) its sink logit.
            v: (H, d) its value.

        Returns:
            (H, d) attention output of the new token over the cached keys and itself.
        """
        f, g, v = f.float(), g.float(), _append_ones(v.float())
        if self.v0 is None:  # first token is the sink
            self.v0, self.n = v, 1
            return self.v0[:, :-1]
        self._insert_block(self._build_block((g[:, 0] - g[:, 1]).unsqueeze(-1), g[:, :1], g[:, 1:], v.unsqueeze(1)))
        self.n += 1
        return self.query(f, b)

    @torch.no_grad()
    def query(self, f: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Attention output (H, d) of query realizers f (H, 2) and sink logit b (H,) over the cached keys."""
        num_heads = f.shape[0]
        width = self.v0.shape[-1]
        qrank = (f[:, 0] - f[:, 1]).unsqueeze(-1)  # (H, 1)
        m_acc = torch.full((num_heads, 1), NEG, device=f.device)
        S_acc = torch.zeros((num_heads, width), device=f.device)
        for block in self.blocks:
            n = block["rank"].shape[-1]
            split = torch.searchsorted(block["rank"].contiguous(), qrank.contiguous(), right=True)  # (H, 1)
            row1, row2 = (split - 1).clamp(min=0), split.clamp(max=n - 1)
            has1, has2 = split > 0, split < n
            logit1 = torch.where(has1, block["m1"].gather(-2, row1.unsqueeze(-1)).squeeze(-1) - f[:, :1], NEG)
            logit2 = torch.where(has2, block["m2"].gather(-2, row2.unsqueeze(-1)).squeeze(-1) - f[:, 1:2], NEG)
            m_acc, S_acc = online_softmax_combine(
                m_acc, S_acc, logit1, block["S1"].gather(-2, row1.unsqueeze(-1).expand(-1, -1, width)).squeeze(-2)
            )
            m_acc, S_acc = online_softmax_combine(
                m_acc, S_acc, logit2, block["S2"].gather(-2, row2.unsqueeze(-1).expand(-1, -1, width)).squeeze(-2)
            )
        m_acc, S_acc = online_softmax_combine(m_acc, S_acc, b.float().unsqueeze(-1), self.v0)
        return S_acc[:, :-1] / S_acc[:, -1:]


_EMPTY = 0xFFFFFFFF  # low word of an empty slot's composite key: above the _sortkey of every non-NaN float


def _chunk_prefix_states(logw: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive prefix states inside each chunk along dim -1 of logw (..., C): ``chunk_scan``'s in-chunk arithmetic.

    The exponentials are taken in fp64 relative to the chunk max and every position is rescaled to its own running max
    before the cast back to fp32. Returns m (..., C, 1), S (..., C, d) for value (..., C, d).
    """
    running_max = torch.cummax(logw, dim=-1).values.unsqueeze(-1)  # running max
    chunk_max = running_max[..., -1:, :]  # chunk max
    chunk_max0 = torch.where(chunk_max == NEG, 0.0, chunk_max).double()
    S = torch.cumsum(torch.exp(logw.double().unsqueeze(-1) - chunk_max0) * value.double(), dim=-2)
    S = torch.where(
        running_max == NEG, 0.0, S * torch.exp(chunk_max0 - torch.where(running_max == NEG, 0.0, running_max).double())
    )
    return running_max, S.float()


def _scan_tree(logw: torch.Tensor, value: torch.Tensor, chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive prefix states, equal to ``chunk_scan`` but with a log-depth carry across chunks.

    Every chunk is scanned on its own (``_chunk_prefix_states``, same fp64 arithmetic as ``chunk_scan``); the chunk
    totals are then scanned by recursive doubling (log2(L / chunk) vectorised combines instead of a Python loop over
    the chunks), and each chunk is combined with the total of all chunks before it. logw: (..., L); value: (..., L, d).
    """
    *lead, length = logw.shape
    d = value.shape[-1]
    chunk_len = min(chunk, length)
    pad = (-length) % chunk_len
    if pad:
        logw = torch.nn.functional.pad(logw, (0, pad), value=NEG)
        value = torch.nn.functional.pad(value, (0, 0, 0, pad))
    nchunks = logw.shape[-1] // chunk_len
    m, S = _chunk_prefix_states(
        logw.reshape(*lead, nchunks, chunk_len), value.reshape(*lead, nchunks, chunk_len, d)
    )  # (..., nchunks, chunk_len, 1), (..., nchunks, chunk_len, d)
    if nchunks > 1:
        m_tot, S_tot = m[..., -1, :], S[..., -1, :]  # chunk totals, (..., nchunks, 1) and (..., nchunks, d)
        k = 1
        while k < nchunks:  # inclusive scan of the totals: after the pass with stride k, state i covers (i-2k, i]
            m_k, S_k = online_softmax_combine(
                m_tot[..., :-k, :], S_tot[..., :-k, :], m_tot[..., k:, :], S_tot[..., k:, :]
            )
            m_tot, S_tot = torch.cat([m_tot[..., :k, :], m_k], dim=-2), torch.cat([S_tot[..., :k, :], S_k], dim=-2)
            k *= 2
        m_excl = m_tot[..., :-1, None, :].expand(*lead, nchunks - 1, chunk_len, 1)  # carry into chunk c: total of < c
        S_excl = S_tot[..., :-1, None, :].expand(*lead, nchunks - 1, chunk_len, d)
        m_c, S_c = online_softmax_combine(m_excl, S_excl, m[..., 1:, :, :], S[..., 1:, :, :])
        m, S = torch.cat([m[..., :1, :, :], m_c], dim=-3), torch.cat([S[..., :1, :, :], S_c], dim=-3)
    m = m.reshape(*lead, nchunks * chunk_len, 1)[..., :length, :]
    S = S.reshape(*lead, nchunks * chunk_len, d)[..., :length, :]
    return m, S


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

    def __init__(
        self,
        chunk: int = 64,
        tail_capacity: int = (1 << 16) - 1,
        min_tail_capacity: int = 1023,
        cuda_graph: bool = False,
        graph_max_level: int = 8,
    ):
        self.chunk = chunk
        self.tail_capacity = tail_capacity
        self.min_tail_capacity = min_tail_capacity
        self.cuda_graph = cuda_graph
        self.graph_max_level = graph_max_level
        self.budget = 1 << 24  # elements (2 x heads x keys x (d+1)) scanned at once: bounds the fp64 temporaries
        self.v0: Optional[torch.Tensor] = None  # (H, d+1) sink value
        self.n = 0  # tokens seen, sink included
        self.ns = 0  # keys in the static block
        self.count = 0  # keys in the tail; bit l set iff tail level l is full
        self.rebuilt = 0  # keys sorted + scanned so far (prefill, merges, folds): the work counter of the tests
        self.folds = 0
        self.captures = 0
        self._graphs: dict[tuple, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}

    # ---------------------------------------------------------------------------------------------------- layout
    def _levels(self, ns: int) -> int:
        """Tail levels L for a static block of ns keys: 2^L - 1 ~ ns / 4, within the capacity bounds."""
        lmax = max(1, self.tail_capacity.bit_length())
        lmin = min(self.min_tail_capacity.bit_length(), lmax)
        return min(lmax, max(lmin, (ns // 4 + 1).bit_length() - 1))

    @staticmethod
    def _slots(level: int) -> tuple[int, int]:
        """Data slots [lo, hi) of tail level ``level``; its guard is slot lo - 1."""
        return (1 << level) + level, (2 << level) + level

    def _alloc(self, num_heads: int, head_dim: int, ns: int, device: torch.device) -> None:
        """(Re)allocate the buffer for a static block of ``ns`` keys and an empty tail; drops the captured graphs."""
        L = self._levels(ns)
        self.L, self.C = L, min(self.tail_capacity, (1 << L) - 1)
        self.base = (1 << L) + L  # first static data slot (its guard is base - 1)
        size = self.base + ns + 1
        logger.debug("OrderCache: allocating %d slots x %d heads (static %d, tail levels %d)", size, num_heads, ns, L)
        lens = torch.tensor([(1 << level) + 1 for level in range(L)] + [ns + 1, 1])  # guard + data per segment
        seg = torch.repeat_interleave(torch.arange(L + 2), lens)
        keys = (seg << 32) | _EMPTY
        guards = torch.cumsum(lens, 0) - lens
        keys[guards] = seg[guards] << 32
        self._empty = keys.unsqueeze(0).to(device)  # (1, size) composite keys of the empty buffer
        self.comp = self._empty.expand(num_heads, -1).clone()
        self.R = torch.zeros(num_heads, size, head_dim + 3, device=device)
        self.P1 = torch.zeros(num_heads, size, head_dim + 2, device=device)
        self.P2 = torch.zeros(num_heads, size, head_dim + 2, device=device)
        self.P1[..., -1] = NEG  # every guard and empty slot is the identity state
        self.P2[..., -1] = NEG
        self.segkey = (torch.arange(L + 1, device=device) << 32).unsqueeze(0)  # (1, L + 1)
        # data slots of levels 0..j-1, read by a carry into level j
        self._below = [
            torch.cat([torch.arange(*self._slots(level)) for level in range(j)]).to(device) if j else None
            for j in range(L)
        ]
        self.ns, self.count = 0, 0
        self._graphs = {}
        self._pool = None  # the graphs of one buffer share a private memory pool; a fresh pool after a reallocation
        self._f_in = torch.zeros(num_heads, 2, device=device)  # static inputs of the CUDA graphs
        self._g_in = torch.zeros(num_heads, 2, device=device)
        self._b_in = torch.zeros(num_heads, device=device)
        self._v_in = torch.zeros(num_heads, head_dim, device=device)
        self._one = torch.ones(num_heads, 1, device=device)

    def nbytes(self) -> int:
        """Bytes of the slot buffers: static block and preallocated tail (no graph pool, no transients)."""
        bufs = (self.comp, self.R, self.P1, self.P2, self.v0)
        return sum(t.numel() * t.element_size() for t in bufs)

    def segments(self) -> list[tuple[int, int]]:
        """Host-side occupancy [(segment, keys)]: the tail levels, then the static block."""
        tail = [(level, (1 << level) if (self.count >> level) & 1 else 0) for level in range(self.L)]
        return tail + [(self.L, self.ns)]

    def _write_segment(self, off: int, seg: int, R: torch.Tensor) -> None:
        """Sort n raw keys by rank and write them with their prefix / suffix states to the slots [off, off + n).

        Args:
            off: first slot.
            seg: the segment id of the slots.
            R: (H, n, d+3) raw keys [v, 1, g1, g2]. Head groups bound the temporaries.
        """
        num_heads, n, width = R.shape
        key = _sortkey(R[..., -2] - R[..., -1])
        if n > 1:
            key, perm = key.sort(dim=-1)
            R = R.gather(1, perm.unsqueeze(-1).expand(-1, -1, width))
        rows = slice(off, off + n)
        self.comp[:, rows] = key | (seg << 32)
        self.R[:, rows] = R
        if n == 1:  # a single key is its own prefix and suffix state: [v, 1, g1] and [v, 1, g2]
            self.P1[:, rows] = R[..., :-1]
            self.P2[:, rows, :-1], self.P2[:, rows, -1] = R[..., :-2], R[..., -1]
            return
        head_group = max(1, self.budget // (2 * n * (width - 2)))
        for h in range(0, num_heads, head_group):
            heads = slice(h, h + head_group)
            v_aug = R[heads, :, :-2]  # [v, 1]
            # prefix scan of exp(g1) and suffix scan (= prefix scan of the reversal) of exp(g2) in one call
            m, S = _scan_tree(
                torch.stack([R[heads, :, -2], R[heads, :, -1].flip(-1)]),
                torch.stack([v_aug, v_aug.flip(-2)]),
                self.chunk,
            )
            self.P1[heads, rows, :-1], self.P1[heads, rows, -1] = S[0], m[0, ..., 0]
            self.P2[heads, rows, :-1], self.P2[heads, rows, -1] = S[1].flip(-2), m[1, ..., 0].flip(-1)

    # ---------------------------------------------------------------------------------------------------- updates
    @torch.no_grad()
    def prefill(self, g: torch.Tensor, value: torch.Tensor) -> None:
        """Resets the cache to a prompt: all N - 1 keys go into one static block, so the tail starts empty and the
        first decode steps rebuild O(1) keys.

        Args:
            g: (H, N, 2) key realizers of positions 0..N-1 (0 = sink).
            value: (H, N, d) values.
        """
        g, value = g.float(), value.float()
        num_heads, seq_len, head_dim = value.shape
        self.v0 = _append_ones(value[:, 0])
        self._alloc(num_heads, head_dim, seq_len - 1, value.device)
        self.n = seq_len
        if seq_len > 1:
            self._write_segment(self.base, self.L, torch.cat([_append_ones(value[:, 1:]), g[:, 1:]], dim=-1))
            self.ns = seq_len - 1
            self.rebuilt += seq_len - 1

    def _merge(self, j: int, g: torch.Tensor, v: torch.Tensor) -> None:
        """Binary-counter carry: rebuild tail level j from the full levels 0..j-1 and the new key g (H, 2), v (H, d).

        The raw keys of the merged levels stay in place (only their composite keys and suffix maxima are reset), so
        running this twice from the same state is harmless, which the CUDA-graph warm-up relies on.
        """
        R = torch.cat([v, self._one, g], dim=-1).unsqueeze(1)
        if j:
            R = torch.cat([self.R.index_select(1, self._below[j]), R], dim=1)
        self._write_segment(self._slots(j)[0], j, R)
        if j:
            end = self._slots(j)[0] - 1  # levels 0..j-1 and their guards: slots [0, end)
            self.comp[:, :end] = self._empty[:, :end]
            self.P2[:, :end, -1] = NEG  # emptied slots read as identity suffix states

    def _fold(self, g: torch.Tensor, v: torch.Tensor) -> None:
        """Fold the tail and the new key into the static block: one rebuild of the cache, then an empty tail."""
        segments = [slice(self.base, self.base + self.ns)]
        segments += [slice(*self._slots(level)) for level in range(self.L) if (self.count >> level) & 1]
        R = torch.cat([self.R[:, s] for s in segments] + [torch.cat([v, self._one, g], dim=-1).unsqueeze(1)], dim=1)
        num_heads, n, width = R.shape
        self.comp = self.R = self.P1 = self.P2 = self._graphs = None  # free the old buffer before reallocating
        self._alloc(num_heads, width - 3, n, R.device)
        self._write_segment(self.base, self.L, R)
        self.ns = n
        self.rebuilt += n
        self.folds += 1

    def _replay(self, key: tuple, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Run ``fn`` from a CUDA graph, capturing it on first use (after one idempotent warm-up run).

        All graphs of the current buffer are captured into ONE private memory pool (one pool per graph held ~13 MiB
        each). Sharing is safe because replays are serialised on one stream and every caller clones the static output
        right after the replay, before another graph can reuse that memory for its temporaries.
        """
        if key not in self._graphs:
            logger.debug("OrderCache: capturing CUDA graph %s", key)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fn()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                out = fn()
            self._pool = graph.pool()
            self._graphs[key] = (graph, out)
            self.captures += 1
        graph, out = self._graphs[key]
        graph.replay()
        return out

    def _use_graphs(self, x: torch.Tensor) -> bool:
        return self.cuda_graph and x.is_cuda

    @torch.no_grad()
    def step(self, f: torch.Tensor, g: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Inserts token n and returns its attention output.

        Args:
            f: (H, 2) query realizers of the new token.
            g: (H, 2) its key realizers.
            b: (H,) its sink logit.
            v: (H, d) its value.

        Returns:
            (H, d) attention output of the new token over the cached keys and itself.
        """
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
        if self._use_graphs(f) and j <= self.graph_max_level:
            torch._foreach_copy_([self._g_in, self._v_in, self._f_in, self._b_in], [g, v, f, b])  # one launch

            def run():
                self._merge(j, self._g_in, self._v_in)
                return self._query(self._f_in, self._b_in)

            return self._replay(("step", j), run).clone()
        self._merge(j, g, v)
        return self.query(f, b)

    def _query(self, f: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        qrank = f[:, 0] - f[:, 1]
        hit = torch.searchsorted(self.comp, self.segkey | _sortkey(qrank).unsqueeze(-1), right=True)  # (H, L + 1)
        width = self.P1.shape[-1]
        P1 = self.P1.gather(1, (hit - 1).unsqueeze(-1).expand(-1, -1, width))  # last branch-1 key of each segment
        P2 = self.P2.gather(1, hit.unsqueeze(-1).expand(-1, -1, width))  # first branch-2 key, or an empty slot / guard
        logit = torch.cat([P1[..., -1] - f[:, :1], P2[..., -1] - f[:, 1:], b.unsqueeze(-1)], dim=-1)
        weights = torch.exp(logit - logit.amax(dim=-1, keepdim=True))  # identity states get weight exactly 0
        S = torch.cat([P1[..., :-1], P2[..., :-1], self.v0.unsqueeze(1)], dim=1)
        out = (weights.unsqueeze(-1) * S).sum(dim=1)
        return out[:, :-1] / out[:, -1:]

    @torch.no_grad()
    def query(self, f: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Attention output (H, d) of query realizers f (H, 2) and sink logit b (H,) over the cached keys."""
        f, b = f.float(), b.float()
        if self._use_graphs(f):
            torch._foreach_copy_([self._f_in, self._b_in], [f, b])
            return self._replay(("query",), lambda: self._query(self._f_in, self._b_in)).clone()
        return self._query(f, b)
