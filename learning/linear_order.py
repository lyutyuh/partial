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
"""
import torch
import triton
import triton.language as tl

# Finite stand-in for -inf inside the scans: (m1 - m) never becomes (-inf) - (-inf).
NEG = -1.0e30
_NEG = tl.constexpr(NEG)  # kernels may only read constexpr globals
MAX_LEN = 1 << 15  # one program holds 2 * next_pow2(N) items in registers; keep sentences/documents below this


@triton.jit
def _lse_combine(m1, s1, m2, s2):
    # (m, s) represents s * exp(m); s may be signed (the backward pass scans upstream gradients).
    m = tl.maximum(m1, m2)
    return m, s1 * tl.exp(m1 - m) + s2 * tl.exp(m2 - m)


@triton.jit
def _argmax_combine(v1, i1, v2, i2):
    take2 = v2 > v1
    return tl.where(take2, v2, v1), tl.where(take2, i2, i1)


@triton.jit
def _merged_order(f1_ptr, f2_ptr, g1_ptr, g2_ptr, base, n, BLOCK: tl.constexpr):
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
    h1 = tl.load(g1_ptr + base + pos, mask=valid & ~is_dep, other=0.0)
    h2 = tl.load(g2_ptr + base + pos, mask=valid & ~is_dep, other=0.0)
    d1 = tl.load(f1_ptr + base + pos, mask=valid & is_dep, other=0.0)
    d2 = tl.load(f2_ptr + base + pos, mask=valid & is_dep, other=0.0)
    key = tl.where(is_dep, d1 - d2, h1 - h2)
    key = tl.where(valid, key, float("inf"))

    # Order-preserving float32 -> uint32 map, packed as [key:32 | is_dep:1 | pos:30] into a non-negative int64.
    bits = key.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    ukey = tl.where(bits >= 0x80000000, (~bits) & 0xFFFFFFFF, bits | 0x80000000)
    packed = (ukey << 31) | (is_dep.to(tl.int64) << 30) | pos.to(tl.int64)
    packed = tl.sort(packed)

    s_dep = ((packed >> 30) & 1) == 1
    s_pos = (packed & 0x3FFFFFFF).to(tl.int32)
    s_ok = s_pos < n
    x1 = tl.where(s_dep, tl.load(f1_ptr + base + s_pos, mask=s_ok & s_dep, other=0.0),
                  tl.load(g1_ptr + base + s_pos, mask=s_ok & ~s_dep, other=0.0))
    x2 = tl.where(s_dep, tl.load(f2_ptr + base + s_pos, mask=s_ok & s_dep, other=0.0),
                  tl.load(g2_ptr + base + s_pos, mask=s_ok & ~s_dep, other=0.0))
    return s_dep, s_pos, s_ok, x1, x2


@triton.jit
def _logz_fwd_kernel(f1_ptr, f2_ptr, g1_ptr, g2_ptr, len_ptr, z_ptr, a_ptr, b_ptr, L,
                     ROOT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    n = tl.load(len_ptr + row)
    base = row.to(tl.int64) * L
    s_dep, s_pos, s_ok, x1, x2 = _merged_order(f1_ptr, f2_ptr, g1_ptr, g2_ptr, base, n, BLOCK)

    head = s_ok & ~s_dep
    one = tl.where(head, 1.0, 0.0)
    pm, ps = tl.associative_scan((tl.where(head, x1, _NEG), one), 0, _lse_combine)                 # heads before
    sm, ss = tl.associative_scan((tl.where(head, x2, _NEG), one), 0, _lse_combine, reverse=True)   # heads after
    a = pm + tl.log(ps) - x1  # log-mass of branch 1; -inf when no head precedes x
    b = sm + tl.log(ss) - x2  # log-mass of branch 2
    mx = tl.maximum(a, b)
    if ROOT:
        mx = tl.maximum(mx, 0.0)
        z = mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx) + tl.exp(-mx))
    else:
        z = mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))

    out = s_ok & s_dep
    tl.store(z_ptr + base + s_pos, z, mask=out)
    tl.store(a_ptr + base + s_pos, a, mask=out)
    tl.store(b_ptr + base + s_pos, b, mask=out)


@triton.jit
def _logz_bwd_kernel(f1_ptr, f2_ptr, g1_ptr, g2_ptr, len_ptr, z_ptr, gz_ptr, dg1_ptr, dg2_ptr, L,
                     BLOCK: tl.constexpr):
    # dZ(x)/dg1(y) = exp(g1(y) - f1(x) - Z(x)) for y before x, so dg1(y) = exp(g1(y)) * sum_{x after y} gZ(x)
    # exp(-f1(x) - Z(x)); symmetrically dg2 sums over the dependents before y. Both are signed log-space scans.
    row = tl.program_id(0)
    n = tl.load(len_ptr + row)
    base = row.to(tl.int64) * L
    s_dep, s_pos, s_ok, x1, x2 = _merged_order(f1_ptr, f2_ptr, g1_ptr, g2_ptr, base, n, BLOCK)

    dep = s_ok & s_dep
    z = tl.load(z_ptr + base + s_pos, mask=dep, other=0.0)
    gz = tl.where(dep, tl.load(gz_ptr + base + s_pos, mask=dep, other=0.0), 0.0)
    rm, rs = tl.associative_scan((tl.where(dep, -x1 - z, _NEG), gz), 0, _lse_combine, reverse=True)
    pm, ps = tl.associative_scan((tl.where(dep, -x2 - z, _NEG), gz), 0, _lse_combine)

    head = s_ok & ~s_dep
    tl.store(dg1_ptr + base + s_pos, rs * tl.exp(rm + x1), mask=head)
    tl.store(dg2_ptr + base + s_pos, ps * tl.exp(pm + x2), mask=head)


@triton.jit
def _decode_kernel(f1_ptr, f2_ptr, g1_ptr, g2_ptr, len_ptr, head_ptr, score_ptr, L,
                   ROOT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    n = tl.load(len_ptr + row)
    base = row.to(tl.int64) * L
    s_dep, s_pos, s_ok, x1, x2 = _merged_order(f1_ptr, f2_ptr, g1_ptr, g2_ptr, base, n, BLOCK)

    head = s_ok & ~s_dep
    idx = tl.where(head, s_pos, -1)
    p1, pi = tl.associative_scan((tl.where(head, x1, _NEG), idx), 0, _argmax_combine)                # max g1 before
    s2, si = tl.associative_scan((tl.where(head, x2, _NEG), idx), 0, _argmax_combine, reverse=True)  # max g2 after
    c1 = p1 - x1
    c2 = s2 - x2
    take2 = c2 > c1
    best = tl.where(take2, c2, c1)
    arg = tl.where(take2, si, pi) + 1  # repo convention: column 0 = ROOT, column j + 1 = word j
    if ROOT:
        root = best <= 0.0  # torch.argmax picks the first column (ROOT) on ties
        best = tl.where(root, 0.0, best)
        arg = tl.where(root, 0, arg)

    out = s_ok & s_dep
    tl.store(head_ptr + base + s_pos, arg, mask=out)
    tl.store(score_ptr + base + s_pos, best, mask=out)


# ----------------------------------------------------------------------------------------------------------------------
# Python API
# ----------------------------------------------------------------------------------------------------------------------

def _split(f, g, lengths):
    if f.shape[-1] != 2 or g.shape != f.shape:
        raise ValueError(f"expected f, g of shape (B, L, 2), got {tuple(f.shape)} and {tuple(g.shape)}")
    if f.shape[1] > MAX_LEN:
        raise ValueError(f"sequence length {f.shape[1]} exceeds MAX_LEN={MAX_LEN}")
    f = f.float()
    g = g.float()
    parts = [t[..., k].contiguous() for t in (f, g) for k in (0, 1)]
    block = max(2, 2 * triton.next_power_of_2(max(int(f.shape[1]), 1)))
    return parts, lengths.to(device=f.device, dtype=torch.int32).contiguous(), block


class _LogPartition(torch.autograd.Function):

    @staticmethod
    def forward(ctx, f, g, lengths, include_root):
        (f1, f2, g1, g2), lens, block = _split(f, g, lengths)
        B, L = f1.shape
        z, a, b = (torch.zeros_like(f1) for _ in range(3))
        _logz_fwd_kernel[(B,)](f1, f2, g1, g2, lens, z, a, b, L, ROOT=include_root, BLOCK=block)
        ctx.save_for_backward(f1, f2, g1, g2, lens, z, a, b)
        ctx.block = block
        ctx.in_dtype = f.dtype
        return z

    @staticmethod
    def backward(ctx, gz):
        f1, f2, g1, g2, lens, z, a, b = ctx.saved_tensors
        gz = gz.float().contiguous()
        dg1, dg2 = torch.zeros_like(g1), torch.zeros_like(g2)
        _logz_bwd_kernel[(f1.shape[0],)](f1, f2, g1, g2, lens, z, gz, dg1, dg2, f1.shape[1], BLOCK=ctx.block)
        valid = _valid_mask(lens, f1.shape[1])
        df1 = torch.where(valid, -gz * torch.exp(a - z), 0.0)
        df2 = torch.where(valid, -gz * torch.exp(b - z), 0.0)
        df = torch.stack([df1, df2], dim=-1).to(ctx.in_dtype)
        dg = torch.stack([dg1, dg2], dim=-1).to(ctx.in_dtype)
        return df, dg, None, None


def _valid_mask(lengths, L):
    return torch.arange(L, device=lengths.device)[None, :] < lengths[:, None]


def log_partition(f, g, lengths, include_root=True):
    """Z(x) = log sum_y exp(-F(x, y)) for every dependent x, in linear memory; differentiable in f and g.

    Args:
        f: Dependent realizer values (``tosets``), shape (B, L, 2).
        g: Head realizer values (``tosets_prime``), shape (B, L, 2).
        lengths: Number of words per sentence, shape (B,); positions >= length are neither heads nor dependents.
        include_root: Add the ROOT head with score 0.

    Returns:
        Z of shape (B, L), fp32, 0 at padding positions.
    """
    return _LogPartition.apply(f, g, lengths, include_root)


@torch.no_grad()
def decode(f, g, lengths, include_root=True):
    """Greedy heads argmax_y -F(x, y) without the (N, N) score matrix.

    Returns:
        heads: (B, L) int32 in the repo's column convention (0 = ROOT, j + 1 = word j); 0 at padding.
        scores: (B, L) fp32 score of the selected head.
    """
    (f1, f2, g1, g2), lens, block = _split(f, g, lengths)
    B, L = f1.shape
    heads = torch.zeros((B, L), dtype=torch.int32, device=f1.device)
    scores = torch.zeros_like(f1)
    _decode_kernel[(B,)](f1, f2, g1, g2, lens, heads, scores, L, ROOT=include_root, BLOCK=block)
    return heads, scores


def gold_arc_score(f, g, heads):
    """s(x, h(x)) for gold heads in the repo convention (0 = ROOT, j + 1 = word j); O(N)."""
    f = f.float()
    g = g.float()
    idx = (heads.clamp(min=1) - 1).long()
    g_head = g.take_along_dim(idx.unsqueeze(-1), dim=1)
    score = -(f - g_head).amax(dim=-1)
    return torch.where(heads == 0, torch.zeros_like(score), score)


def arc_loss(f, g, heads, lengths, include_root=True):
    """Linear-memory arc cross-entropy: mean over words of Z(x) - s(x, h(x)); heads = -1 marks padding."""
    valid = (heads >= 0) & _valid_mask(lengths.to(heads.device), heads.shape[1])
    nll = log_partition(f, g, lengths, include_root) - gold_arc_score(f, g, heads.clamp(min=0))
    return nll[valid].mean()


# ----------------------------------------------------------------------------------------------------------------------
# Pure-torch references
# ----------------------------------------------------------------------------------------------------------------------

def arc_scores_quadratic(f, g, lengths, include_root=True):
    """The O(N^2) score matrix with the hard max of Eq. 2: (B, L, L + 1) with ROOT in column 0 (-inf if excluded)."""
    f = f.float()
    g = g.float()
    s = -(f.unsqueeze(2) - g.unsqueeze(1)).amax(dim=-1)
    head_ok = _valid_mask(lengths.to(f.device), f.shape[1]).unsqueeze(1)
    s = s.masked_fill(~head_ok, float("-inf"))
    root = s.new_zeros(s.shape[:2] + (1,)) if include_root else s.new_full(s.shape[:2] + (1,), float("-inf"))
    return torch.cat([root, s], dim=-1)


def log_partition_torch(f, g, lengths, include_root=True):
    """O(N log N) pure-torch Alg. 1 (sort + logcumsumexp + searchsorted); a CPU fallback and second reference."""
    f = f.float()
    g = g.float()
    B, L, _ = f.shape
    head_ok = _valid_mask(lengths.to(f.device), L)
    e = torch.where(head_ok, g[..., 0] - g[..., 1], float("inf"))
    e_sorted, order = e.sort(dim=-1)
    gs = g.take_along_dim(order.unsqueeze(-1), dim=1)
    gs = torch.where(head_ok.gather(1, order).unsqueeze(-1), gs, NEG)
    neg = gs.new_full((B, 1), float("-inf"))
    pre1 = torch.cat([neg, torch.logcumsumexp(gs[..., 0], dim=-1)], dim=-1)               # heads sorted [0, r)
    suf2 = torch.cat([torch.logcumsumexp(gs[..., 1].flip(-1), dim=-1).flip(-1), neg], dim=-1)  # heads sorted [r, L)
    r = torch.searchsorted(e_sorted.contiguous(), (f[..., 0] - f[..., 1]).contiguous(), right=True)
    terms = [pre1.gather(1, r) - f[..., 0], suf2.gather(1, r) - f[..., 1]]
    if include_root:
        terms.append(torch.zeros_like(terms[0]))
    z = torch.logsumexp(torch.stack(terms, dim=-1), dim=-1)
    return torch.where(head_ok, z, 0.0)
