"""Linear-time partial-order arc aggregation (Liu et al., EMNLP 2023, Sec. 4.3, Alg. 1) as Triton kernels.

Arc scores follow ``ModelForPartialOrder``: a dependent x has realizer values f(x) = (f1, f2) (``tosets``), a candidate
head y has g(y) = (g1, g2) (``tosets_prime``), and

    s(x, y) = -F(x, y),    F(x, y) = max(f1(x) - g1(y), f2(x) - g2(y))           (Eq. 2, K = 2, hard max)

plus an optional ROOT head with score 0 (the zero column the model prepends). The model materialises the (N, N) score
matrix; here every per-dependent aggregation is done without it:

    log_partition  Z(x) = log sum_y exp(s(x, y))     (the softmax normaliser of the arc cross-entropy)
    decode         argmax_y s(x, y)                  (greedy head selection)

Fredman's trick: F(x, y) = f1(x) - g1(y) iff g1(y) - g2(y) <= f1(x) - f2(x). Sorting heads and dependents together by
those keys, every dependent's head set splits into the heads before it (branch 1) and after it (branch 2), so

    Z(x) = logaddexp(LSE_{y before x} g1(y) - f1(x),  LSE_{y after x} g2(y) - f2(x)  [, 0 for ROOT])

is one forward and one reverse scan over the merged order; the backward pass is the same two scans run over the
dependents in the opposite directions. One Triton program handles one sentence: a bitonic ``tl.sort`` of 2N packed
keys (O(N log^2 N) work, O(log^2 N) depth) followed by O(N)-work ``tl.associative_scan``s. Accumulation is fp32.

Layout: f, g (batch, seqlen, 2); lengths (batch,). Kernels take the four realizer columns as separate contiguous
(batch, seqlen) arrays F1, F2, G1, G2.
"""
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Finite stand-in for -inf inside the scans: (m1 - m) never becomes (-inf) - (-inf).
NEG = -1.0e30
_NEG = tl.constexpr(NEG)  # kernels may only read constexpr globals
MAX_LEN = 1 << 13  # one program sorts 2 * next_pow2(N) items; above 8192 the GH200 runs out of shared memory


# ----------------------------------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------------------------------


@triton.jit
def _logsumexp_combine(m1, s1, m2, s2):
    # (m, s) represents s * exp(m); s may be signed (the backward pass scans upstream gradients).
    m = tl.maximum(m1, m2)
    return m, s1 * tl.exp(m1 - m) + s2 * tl.exp(m2 - m)


@triton.jit
def _argmax_combine(v1, i1, v2, i2):
    take2 = v2 > v1
    return tl.where(take2, v2, v1), tl.where(take2, i2, i1)


@triton.jit
def _merged_order(F1, F2, G1, G2, base, n, BLOCK: tl.constexpr):
    """Sort the n heads (key g1 - g2) and n dependents (key f1 - f2) of one sentence into a single ascending order.

    Returns, per sorted slot: is-dependent flag, original position, validity, and the slot's two realizer values
    (f1, f2 for a dependent, g1, g2 for a head). At equal keys heads come first, so a dependent's prefix holds exactly
    the heads with g1 - g2 <= f1 - f2. Padding slots sort to the end.
    """
    HALF: tl.constexpr = BLOCK // 2
    offs = tl.arange(0, BLOCK)
    is_dep = offs >= HALF
    pos = tl.where(is_dep, offs - HALF, offs)
    valid = pos < n
    g1_h = tl.load(G1 + base + pos, mask=valid & ~is_dep, other=0.0)
    g2_h = tl.load(G2 + base + pos, mask=valid & ~is_dep, other=0.0)
    f1_d = tl.load(F1 + base + pos, mask=valid & is_dep, other=0.0)
    f2_d = tl.load(F2 + base + pos, mask=valid & is_dep, other=0.0)
    key = tl.where(is_dep, f1_d - f2_d, g1_h - g2_h)
    key = tl.where(valid, key, float("inf"))
    key = tl.where(key == 0.0, 0.0, key)  # -0.0 -> +0.0, so equal keys always put heads first

    # Order-preserving float32 -> uint32 map, packed as [key:32 | is_dep:1 | pos:30] into a non-negative int64.
    bits = key.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    ukey = tl.where(bits >= 0x80000000, (~bits) & 0xFFFFFFFF, bits | 0x80000000)
    packed = (ukey << 31) | (is_dep.to(tl.int64) << 30) | pos.to(tl.int64)
    packed = tl.sort(packed)

    sorted_is_dep = ((packed >> 30) & 1) == 1
    sorted_pos = (packed & 0x3FFFFFFF).to(tl.int32)
    sorted_ok = sorted_pos < n
    x1 = tl.where(
        sorted_is_dep,
        tl.load(F1 + base + sorted_pos, mask=sorted_ok & sorted_is_dep, other=0.0),
        tl.load(G1 + base + sorted_pos, mask=sorted_ok & ~sorted_is_dep, other=0.0),
    )
    x2 = tl.where(
        sorted_is_dep,
        tl.load(F2 + base + sorted_pos, mask=sorted_ok & sorted_is_dep, other=0.0),
        tl.load(G2 + base + sorted_pos, mask=sorted_ok & ~sorted_is_dep, other=0.0),
    )
    return sorted_is_dep, sorted_pos, sorted_ok, x1, x2


@triton.jit
def _logz_fwd_kernel(
    F1,
    F2,
    G1,
    G2,
    Lens,
    Z,
    A,
    B,
    seqlen,
    INCLUDE_ROOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Z(x) of every dependent of one sentence, plus the branch log-masses A (heads before x) and B (heads after)."""
    row = tl.program_id(0)
    n = tl.load(Lens + row)
    base = row.to(tl.int64) * seqlen
    is_dep, pos, ok, x1, x2 = _merged_order(F1, F2, G1, G2, base, n, BLOCK)

    head = ok & ~is_dep
    one = tl.where(head, 1.0, 0.0)
    m_pre, s_pre = tl.associative_scan((tl.where(head, x1, _NEG), one), 0, _logsumexp_combine)  # heads before
    m_suf, s_suf = tl.associative_scan((tl.where(head, x2, _NEG), one), 0, _logsumexp_combine, reverse=True)  # after
    logmass1 = (m_pre - x1) + tl.log(s_pre)  # log-mass of branch 1; -inf when no head precedes x
    logmass2 = (m_suf - x2) + tl.log(s_suf)  # log-mass of branch 2 (differences first: rounds at |g - f|, not |g|)
    m = tl.maximum(logmass1, logmass2)
    if INCLUDE_ROOT:
        m = tl.maximum(m, 0.0)
        z = m + tl.log(tl.exp(logmass1 - m) + tl.exp(logmass2 - m) + tl.exp(-m))
    else:
        z = m + tl.log(tl.exp(logmass1 - m) + tl.exp(logmass2 - m))

    out = ok & is_dep
    tl.store(Z + base + pos, z, mask=out)
    tl.store(A + base + pos, logmass1, mask=out)
    tl.store(B + base + pos, logmass2, mask=out)


@triton.jit
def _logz_bwd_kernel(
    F1,
    F2,
    G1,
    G2,
    Lens,
    Z,
    DZ,
    DG1,
    DG2,
    seqlen,
    BLOCK: tl.constexpr,
):
    """dg1, dg2 of every head of one sentence from the upstream gradient DZ of Z.

    dZ(x)/dg1(y) = exp(g1(y) - f1(x) - Z(x)) for y before x, so dg1(y) = exp(g1(y)) * sum_{x after y} dZ(x)
    exp(-f1(x) - Z(x)); symmetrically dg2 sums over the dependents before y. Both are signed log-space scans.
    """
    row = tl.program_id(0)
    n = tl.load(Lens + row)
    base = row.to(tl.int64) * seqlen
    is_dep, pos, ok, x1, x2 = _merged_order(F1, F2, G1, G2, base, n, BLOCK)

    dep = ok & is_dep
    z = tl.load(Z + base + pos, mask=dep, other=0.0)
    dz = tl.where(dep, tl.load(DZ + base + pos, mask=dep, other=0.0), 0.0)
    m_suf, s_suf = tl.associative_scan((tl.where(dep, -x1 - z, _NEG), dz), 0, _logsumexp_combine, reverse=True)
    m_pre, s_pre = tl.associative_scan((tl.where(dep, -x2 - z, _NEG), dz), 0, _logsumexp_combine)

    head = ok & ~is_dep
    tl.store(DG1 + base + pos, s_suf * tl.exp(m_suf + x1), mask=head)
    tl.store(DG2 + base + pos, s_pre * tl.exp(m_pre + x2), mask=head)


@triton.jit
def _decode_kernel(
    F1,
    F2,
    G1,
    G2,
    Lens,
    Heads,
    Scores,
    seqlen,
    INCLUDE_ROOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Greedy head (argmax_y s(x, y)) and its score of every dependent of one sentence."""
    row = tl.program_id(0)
    n = tl.load(Lens + row)
    base = row.to(tl.int64) * seqlen
    is_dep, pos, ok, x1, x2 = _merged_order(F1, F2, G1, G2, base, n, BLOCK)

    head = ok & ~is_dep
    idx = tl.where(head, pos, -1)
    best1, idx1 = tl.associative_scan((tl.where(head, x1, _NEG), idx), 0, _argmax_combine)  # max g1 before
    best2, idx2 = tl.associative_scan((tl.where(head, x2, _NEG), idx), 0, _argmax_combine, reverse=True)  # max g2 after
    score1 = best1 - x1
    score2 = best2 - x2
    take2 = score2 > score1
    best = tl.where(take2, score2, score1)
    arg = tl.where(take2, idx2, idx1) + 1  # repo convention: column 0 = ROOT, column j + 1 = word j
    if INCLUDE_ROOT:
        root = best <= 0.0  # torch.argmax picks the first column (ROOT) on ties
        best = tl.where(root, 0.0, best)
        arg = tl.where(root, 0, arg)

    out = ok & is_dep
    tl.store(Heads + base + pos, arg, mask=out)
    tl.store(Scores + base + pos, best, mask=out)


# ----------------------------------------------------------------------------------------------------------------------
# Python API
# ----------------------------------------------------------------------------------------------------------------------


def _prepare_inputs(
    f: torch.Tensor, g: torch.Tensor, lengths: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor, int]:
    """Checks the shapes and returns ([f1, f2, g1, g2] fp32 contiguous, int32 lengths, the sort block size)."""
    if f.shape[-1] != 2 or g.shape != f.shape:
        raise ValueError(f"expected f, g of shape (batch, seqlen, 2), got {tuple(f.shape)} and {tuple(g.shape)}")
    if f.shape[1] > MAX_LEN:
        raise ValueError(f"sequence length {f.shape[1]} exceeds MAX_LEN={MAX_LEN}")
    if int(lengths.max()) > f.shape[1]:
        raise ValueError(f"lengths up to {int(lengths.max())} exceed the padded length {f.shape[1]}")
    f = f.float()
    g = g.float()
    parts = [t[..., k].contiguous() for t in (f, g) for k in (0, 1)]
    block_size = max(2, 2 * triton.next_power_of_2(max(int(f.shape[1]), 1)))
    return parts, lengths.to(device=f.device, dtype=torch.int32).contiguous(), block_size


def _valid_mask(lengths: torch.Tensor, seqlen: int) -> torch.Tensor:
    return torch.arange(seqlen, device=lengths.device)[None, :] < lengths[:, None]


class LogPartitionFunc(torch.autograd.Function):
    """Z with the Triton forward and backward scans; df follows from the saved branch log-masses."""

    @staticmethod
    def forward(ctx, f, g, lengths, include_root):
        (f1, f2, g1, g2), lens, block_size = _prepare_inputs(f, g, lengths)
        batch, seqlen = f1.shape
        z, a, b = (torch.zeros_like(f1) for _ in range(3))
        _logz_fwd_kernel[(batch,)](f1, f2, g1, g2, lens, z, a, b, seqlen, INCLUDE_ROOT=include_root, BLOCK=block_size)
        ctx.save_for_backward(f1, f2, g1, g2, lens, z, a, b)
        ctx.block_size = block_size
        ctx.in_dtype = f.dtype
        return z

    @staticmethod
    def backward(ctx, dz):
        f1, f2, g1, g2, lens, z, a, b = ctx.saved_tensors
        dz = dz.float().contiguous()
        dg1, dg2 = torch.zeros_like(g1), torch.zeros_like(g2)
        _logz_bwd_kernel[(f1.shape[0],)](f1, f2, g1, g2, lens, z, dz, dg1, dg2, f1.shape[1], BLOCK=ctx.block_size)
        valid = _valid_mask(lens, f1.shape[1])
        df1 = torch.where(valid, -dz * torch.exp(a - z), 0.0)
        df2 = torch.where(valid, -dz * torch.exp(b - z), 0.0)
        df = torch.stack([df1, df2], dim=-1).to(ctx.in_dtype)
        dg = torch.stack([dg1, dg2], dim=-1).to(ctx.in_dtype)
        return df, dg, None, None


def log_partition(f: torch.Tensor, g: torch.Tensor, lengths: torch.Tensor, include_root: bool = True) -> torch.Tensor:
    """Z(x) = log sum_y exp(-F(x, y)) for every dependent x, in linear memory; differentiable in f and g.

    Args:
        f: Dependent realizer values (``tosets``), shape (batch, seqlen, 2).
        g: Head realizer values (``tosets_prime``), shape (batch, seqlen, 2).
        lengths: Number of words per sentence, shape (batch,); positions >= length are neither heads nor dependents.
        include_root: Add the ROOT head with score 0.

    Returns:
        Z of shape (batch, seqlen), fp32, 0 at padding positions.
    """
    return LogPartitionFunc.apply(f, g, lengths, include_root)


@torch.no_grad()
def decode(
    f: torch.Tensor, g: torch.Tensor, lengths: torch.Tensor, include_root: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy heads argmax_y -F(x, y) without the (N, N) score matrix.

    Args:
        f, g, lengths, include_root: as in ``log_partition``.

    Returns:
        heads: (batch, seqlen) int32 in the repo's column convention (0 = ROOT, j + 1 = word j); 0 at padding.
        scores: (batch, seqlen) fp32 score of the selected head.
    """
    (f1, f2, g1, g2), lens, block_size = _prepare_inputs(f, g, lengths)
    batch, seqlen = f1.shape
    heads = torch.zeros((batch, seqlen), dtype=torch.int32, device=f1.device)
    scores = torch.zeros_like(f1)
    _decode_kernel[(batch,)](f1, f2, g1, g2, lens, heads, scores, seqlen, INCLUDE_ROOT=include_root, BLOCK=block_size)
    return heads, scores


def gold_arc_score(f: torch.Tensor, g: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    """s(x, h(x)) for gold heads in the repo convention (0 = ROOT, j + 1 = word j); O(N)."""
    f = f.float()
    g = g.float()
    idx = (heads.clamp(min=1) - 1).long()
    g_head = g.take_along_dim(idx.unsqueeze(-1), dim=1)
    # same branch rule as the kernels, so exact F-ties send the gold gradient where log_partition sends it
    branch1 = (g_head[..., 0] - g_head[..., 1]) <= (f[..., 0] - f[..., 1])
    score = torch.where(branch1, g_head[..., 0] - f[..., 0], g_head[..., 1] - f[..., 1])
    return torch.where(heads == 0, torch.zeros_like(score), score)


def arc_loss(
    f: torch.Tensor, g: torch.Tensor, heads: torch.Tensor, lengths: torch.Tensor, include_root: bool = True
) -> torch.Tensor:
    """Linear-memory arc cross-entropy: mean over words of Z(x) - s(x, h(x)); heads = -1 marks padding."""
    valid = (heads >= 0) & _valid_mask(lengths.to(heads.device), heads.shape[1])
    nll = log_partition(f, g, lengths, include_root) - gold_arc_score(f, g, heads.clamp(min=0))
    return nll[valid].mean()


# ----------------------------------------------------------------------------------------------------------------------
# Pure-torch references
# ----------------------------------------------------------------------------------------------------------------------


def arc_scores_quadratic(
    f: torch.Tensor, g: torch.Tensor, lengths: torch.Tensor, include_root: bool = True
) -> torch.Tensor:
    """The O(N^2) score matrix with the hard max of Eq. 2: (batch, seqlen, seqlen + 1) with ROOT in column 0 (-inf if
    excluded)."""
    f = f.float()
    g = g.float()
    scores = -(f.unsqueeze(2) - g.unsqueeze(1)).amax(dim=-1)
    head_ok = _valid_mask(lengths.to(f.device), f.shape[1]).unsqueeze(1)
    scores = scores.masked_fill(~head_ok, float("-inf"))
    if include_root:
        root = scores.new_zeros(scores.shape[:2] + (1,))
    else:
        root = scores.new_full(scores.shape[:2] + (1,), float("-inf"))
    return torch.cat([root, scores], dim=-1)


def log_partition_torch(
    f: torch.Tensor, g: torch.Tensor, lengths: torch.Tensor, include_root: bool = True
) -> torch.Tensor:
    """O(N log N) pure-torch Alg. 1 (sort + logcumsumexp + searchsorted); a CPU fallback and second reference."""
    f = f.float()
    g = g.float()
    batch, seqlen, _ = f.shape
    head_ok = _valid_mask(lengths.to(f.device), seqlen)
    krank = torch.where(head_ok, g[..., 0] - g[..., 1], float("inf"))
    krank_sorted, order = krank.sort(dim=-1)
    g_sorted = g.take_along_dim(order.unsqueeze(-1), dim=1)
    g_sorted = torch.where(head_ok.gather(1, order).unsqueeze(-1), g_sorted, NEG)
    neg = g_sorted.new_full((batch, 1), float("-inf"))
    prefix1 = torch.cat([neg, torch.logcumsumexp(g_sorted[..., 0], dim=-1)], dim=-1)  # heads sorted [0, r)
    suffix2 = torch.cat([torch.logcumsumexp(g_sorted[..., 1].flip(-1), dim=-1).flip(-1), neg], dim=-1)  # [r, L)
    split = torch.searchsorted(krank_sorted.contiguous(), (f[..., 0] - f[..., 1]).contiguous(), right=True)
    terms = [prefix1.gather(1, split) - f[..., 0], suffix2.gather(1, split) - f[..., 1]]
    if include_root:
        terms.append(torch.zeros_like(terms[0]))
    z = torch.logsumexp(torch.stack(terms, dim=-1), dim=-1)
    return torch.where(head_ok, z, 0.0)
