"""Causal order attention (learning/order_attention.py) as Triton kernels: a dense intra-chunk tile plus a dyadic tree
over whole chunks, like flash attention's causal split.

Per (batch, head): query realizers f(x) = (f1, f2), key realizers g(y) = (g1, g2), sink logit b(x), values v(y), and

    logit(x, y) = min(g1(y) - f1(x), g2(y) - f2(x))   for 1 <= y <= x,      logit(x, 0) = b(x)   (key 0 is the sink).

With query rank a(x) = f1(x) - f2(x) and key rank r(y) = g1(y) - g2(y), y is on branch 1 of x iff r(y) <= a(x) (logit
g1(y) - f1(x)), else on branch 2 (logit g2(y) - f2(x)). The kernels decide every branch by this one fp32 comparison of
the two ranks (also inside the dense tile), so the forward and a backward always agree on it; at a near-tie the two
candidate logits differ only by rounding.

Positions are cut into nc = ceil(N / C) chunks of C keys. A query x in chunk c sees

* INTRA: the keys y <= x of its own chunk (key 0 excluded), scored densely; P @ V on ``tl.dot`` in fp32.
* INTER: every key of chunks 0 .. c-1, as one dyadic block per set bit l of c: the 2^l chunks starting at chunk
  (c >> (l + 1)) << (l + 1). Only even-indexed level-l blocks are ever read, and only those with a query chunk after
  them, so level l (l = 0 .. ceil(log2 nc) - 1) holds nb_l = (nc - 2^l - 1) // 2^(l+1) + 1 blocks of Lb_l = C 2^l keys
  (about N / 2 keys per level). No block contains the last chunk, so the tree never sees padding positions.

Kernels (one forward = ceil(log2 nc) segmented ``torch.sort`` calls + 3 Triton launches):

1. Sort: per level, ``torch.sort`` of the key ranks along the last dim of a (BH, nb_l, Lb_l) view (a segmented sort;
   equivalent to sorting the composite keys (block << 32 | rank) of ``order_attention._sort_blocks``).
2. ``_scan_kernel`` (all levels in one launch, top level first): at every sorted slot i of a block, the inclusive PREFIX
   state of exp(g1) [v, 1] (slots <= i) and the inclusive SUFFIX state of exp(g2) [v, 1] (slots >= i). A state
   (m, S, z) is relative to its reference m = ceil(own running max), a whole number of nats: S = sum exp(g - m) v,
   z = sum exp(g - m). Within a tile of T sorted keys this is the linear recurrence S_i = exp(m_{i-1} - m_i) S_{i-1} +
   exp(g_i - m_i) v_i, run by ``tl.associative_scan`` with an affine combine, plus a sequential carry across tiles
   rescaled to each slot by one factor exp(m_carry - m_i) and summed with a TwoSum compensation. Every exponent is
   <= 0, so nothing overflows; a term only underflows when it is below 2^-126 of its state's reference; and a rescale
   factor is exactly 1 unless the running max crosses a whole nat, so neither the rescale roundings nor the carry
   additions accumulate with the block length (relative to the running max itself, the rounding of the per-record
   factors grew linearly with it: lse / A1 errors of 9x the 1e-5 tolerance at N = 2^20 for a constant realizer).
3. ``_search_kernel``: r(x, l) = #keys of x's level-l block with rank <= a(x), a branchless binary search over a
   (queries, levels) tile. (Deviation from a search fused into the query kernel, measured at N = 32768, H = 16,
   d = 128: fused, the chain of dependent loads ran at the query kernel's low occupancy and cost 1.85 ms; split, the
   search takes 0.07 ms and the query kernel 1.55 ms.)
4. ``_query_kernel``, one program per (BLOCK_Q queries, b * H + h, d-slice): for each set bit l of the chunk, the
   prefix state at slot r - 1 shifted by -f1(x) and the suffix state at slot r shifted by -f2(x) are combined online
   into per-branch register accumulators; then the intra tile and the sink exp(b(x)) [v(0), 1].

Cost O(N C d) intra + O(N d log(N / C)) inter. The level buffers are transient (freed after the forward). Layout, from
``_build_levels``: one flat row space, level-major, rows ordered (level l, bh, block k, sorted slot) and level l
starting at row lvl[l] = BH sum_{l' < l} nb_l' Lb_l' (computed inside the kernels, so no offset table is copied or
cached); ``rank`` (rows,) fp32 sorted key ranks, ``perm`` (rows,) int64 key offsets inside the block, ``S1`` / ``S2``
(rows, d) fp32 prefix / suffix value sums, ``mz1`` / ``mz2`` (rows, 2) fp32 [reference m, normaliser z]; plus ``r``
(B H, nlev, N) int32 from the search. Per level and head that is about N / 2 rows of 2 (d + 2) + 3 words: 2.9 GB at
N = 32768, H = 16, d = 128 (bf16 SDPA: 0.13 GB).

Measured on a GH200 (scripts/bench_order_attention_triton.py, B = 1, H = 16, d = 128, fp32, chunk 64): 3.9 ms at
N = 32768 and 8.1 ms at N = 65536 (bf16 values: 3.5 / 7.4 ms), against 8.5 / 34.4 ms for the fastest bf16 causal SDPA
forward, cuDNN (FlashAttention-2, the default dispatch: 15.5 / 61.4 ms); break-even with cuDNN near N = 16384 (1.90 vs
1.80 ms), and at N = 8192 it is 1.9x slower (0.98 vs 0.52 ms).

Saved for the backward (all fp32, contiguous, returned by ``_order_attention_fwd``):

* ``out`` (B, H, N, d): the attention output.
* ``lse`` (B, H, N): log sum_y exp(logit(x, y)) over the sink and the visible keys, so p(x, y) = exp(logit - lse).
* ``A1`` (B, H, N): branch-1 probability mass sum_{y on branch 1} p(x, y) (intra and inter keys).
* ``U1`` (B, H, N, d): branch-1 output sum_{y on branch 1} p(x, y) v(y) (intra and inter keys).
* ``p0`` (B, H, N): sink probability p(x, 0) = exp(b(x) - lse(x)).
* ``lerr`` (B, H, N): the rounding residual of lse = m + log z (TwoSum), so that lse + lerr is m + log z exactly.

Branch 2 follows as A2 = 1 - A1 - p0, U2 = out - U1 - p0 v(0). Query 0 sees only the sink: out(0) = v(0), p0(0) = 1.

Backward (``_order_attention_bwd``). With dO the upstream gradient, D(x) = <dO(x), out(x)> and ds(x, y) =
p(x, y) (<dO(x), v(y)> - D(x)) (the softmax backward):

* Per query, elementwise from the saved tensors (``_bwd_prep_kernel``): df1 = -(<dO, U1> - D A1), df2 = -(<dO, U2> -
  D A2), db = p0 (<dO, v(0)> - D); and dV(0) = sum_x p0(x) dO(x), as key 0 is only ever the sink.
* Per key y >= 1: dV(y) = e^{g1(y)} T1(y) + e^{g2(y)} T2(y) and dg_k(y) = e^{g_k(y)} (<v(y), T_k(y)> - T_kD(y)), with
  [T_k, T_kD](y) = sum of e^{-f_k(x) - lse(x)} [dO(x), D(x)] over the queries x >= y that put y on branch k (branch 1:
  a(x) >= r(y), branch 2: a(x) < r(y), the forward's tie rule). That is the transposed problem, solved by the same
  chunked split with queries and keys swapped:

  1. Sort: the queries of every ODD level block by a(x), the blocks read by the keys of the even block before them
     (``_bwd_build_levels``; same row layout, blocks of the last level may run past N: padding sorts last).
  2. ``_bwd_scan_kernel``: the branch-1 SUFFIX and branch-2 PREFIX states of [dO, D] (signed) with log-weights
     -f1 - lse and -f2 - lse, numerically as ``_scan_kernel``.
  3. ``_bwd_search_kernel``: s(y, l) = #queries with a(x) < r(y) in the level-l block read by key y.
  4. ``_key_intra_kernel``: the dense tile of the key's own chunk, flash-attention style in the keys x queries
     orientation (P^T = exp(logit - lse), dP^T = V dO^T, dS^T = P^T (dP^T - D); dV = P^T dO, dg_k = branch-k row sums).
  5. ``_key_tree_kernel``: adds exp(g_k(y) + m) [S, <v(y), S> - zD] of the state at slot s (branch 1) and s - 1
     (branch 2) of every level. A linear sum suffices: every term is a probability times an upstream quantity, and
     the exponent is log p(x, y) of the state's largest query up to the whole-nat rounding of m.

  Probabilities are recomputed from lse, and the fp32 lse rounds by |lse| 2^-24: p = exp(logit - lse) alone was off by
  up to 1e-5 relative at |lse| ~ 200 (scale-50 integer inputs, whose fp32 logits are exact: dV errors of 3e-6 of the
  gradient scale, 4x the fp32 dense reference). So the forward also saves lse's TwoSum residual, the intra tile uses
  exp((logit - lse) - lerr) and the scan carries -f - lse as a hi / lo pair; after that dV is at 1.4e-7 of the scale.
  The tree's dg forms <v(y), sum_x p dO(x)> - sum_x p D(x) after the sum over queries (the dense softmax backward
  cancels per pair), which leaves dg errors up to ~1.5e-6 of the gradient scale at large magnitudes.

Backward cost: like the forward, O(N C d) + O(N d log(N / C)) time and the same transient level buffers. Measured on a
GH200 (same setting as above): backward kernels 4.45 ms at N = 32768 (prep 0.24, sorts 0.29, scan 1.77, search 0.07,
intra 1.28, tree 0.70, glue 0.10); forward + backward 8.5 / 17.6 ms at N = 32768 / 65536 against 33.9 / 132.4 ms for
cuDNN bf16 SDPA (4.0x / 7.5x), 2.0x at N = 16384 (4.1 vs 8.3 ms) and about even at N = 8192 (2.1-2.9 ms over runs,
mostly host launch overhead, vs 2.0 ms).

Mixture (``mixture_attention_triton``): out = sum_m w_m(x) out_m(x) with w = softmax(gates) over M components sharing
V, as one autograd function (``_MixtureAttention``) rather than autograd over the per-component outputs. The forward
writes every component's output into one (M, B, H, N, d) buffer and forms the weighted sum in one pass
(``_mix_sum_kernel``). Component m's upstream gradient is w_m dO; instead of forming it, its backward takes dO itself:
the prep kernel (``HAS_W``) multiplies df, db and the dV(0) parts by w_m, writes dL/dw_m = <dO, out_m>, and hands the
scan and intra kernels lerr - log w_m in place of the lse residual lerr, so every probability they recompute is w_m p
(exactly 0 for w_m = 0, where that is +inf). That costs those two kernels no extra load: multiplying by a gathered w_m
in the scan and the intra tile instead was 0.45 ms slower at N = 32768. The fold rounds by about |log w_m| 2^-23, a
relative error of that size in component m's probabilities, which are scaled by w_m: at most max_w w |log w| 2^-23
~ 5e-8 of the unweighted gradient scale. All components add into one dV (the intra kernel with ``ACC``, added after
its loop), and dgates follows through the softmax Jacobian. Without gradients (``torch.no_grad``, or no input
requiring grad) each component's output is added into the result and dropped, so only one component's tensors are
alive at a time (eval memory as for M = 1 plus one output). Measured (GH200, B = 1, H = 16, d = 128, M = 4, chunk
64): forward + backward 8.4 / 34.2 / 72.0 ms at N = 8192 / 32768 / 65536 against 9.5 / 38.2 / 80.2 ms for autograd
over the per-component outputs (whose elementwise glue was 5.4 of its 37.4 ms of kernels at N = 32768; now 1.1 ms
including the weighted sum), i.e. even with cuDNN bf16 SDPA at N = 32768 (34.2 ms); peak memory 4.8 vs 5.0 GB there.
No-grad forward 4.0 / 16.2 ms at N = 8192 / 32768 (autograd version 4.2 / 17.0 ms) at the same peak memory.
"""
import contextlib
import math
import os

import torch
import triton
import triton.language as tl

# Finite stand-in for -inf: (m_a - m_b) never becomes (-inf) - (-inf).
NEG = -1.0e30
_NEG = tl.constexpr(NEG)  # kernels may only read constexpr globals


# ----------------------------------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------------------------------

@triton.jit
def _max_incl_excl(i1, e1, i2, e2):
    # A segment is (max over it, max over it without its last element); combining keeps both.
    return tl.maximum(i1, i2), tl.maximum(i1, e2)


@triton.jit
def _affine(a1, b1, a2, b2):
    # Composition of h -> a1 h + b1, then h -> a2 h + b2.
    return a1 * a2, b1 * a2 + b2


@triton.jit
def _level_row(lev, BH, nc, NLEV, C: tl.constexpr):
    """First buffer row of level ``lev`` (scalar or vector): BH C sum_{l < lev} nb_l 2^l, computed on the device."""
    off = tl.zeros_like(lev).to(tl.int64)
    for li in range(0, NLEV):
        nb_l = ((nc - (1 << li) - 1) >> (li + 1)) + 1
        off += tl.where(li < lev, nb_l << li, 0).to(tl.int64)
    return off * BH * C


@triton.jit
def _scan_kernel(perm_ptr, g_ptr, v_ptr, S1_ptr, mz1_ptr, S2_ptr, mz2_ptr,
                 N, BH, nc, NLEV,
                 C: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, NDS: tl.constexpr, T: tl.constexpr):
    """Prefix (direction 0) or suffix (direction 1) states of one rank-sorted level block, for one d-slice.

    One launch covers every level. Program ids run over (task, bh, direction, d-slice) with the task slowest; tasks are
    the level blocks, top level first, so the long sequential scans of the big blocks start first and the many short
    ones fill the machine behind them. perm holds, per sorted slot, the key's offset inside its block (block k of level
    l starts at position k 2^(l+1) C); the states go to the same rows as perm: lvl[l] + (bh nb_l + k) Lb_l + slot.

    The state of slot i is kept relative to its reference m_i = ceil(running max through slot i), a whole number of
    nats: m_i >= every g in it (no exponent above 0) and m_i < running max + 1 (at most one nat more underflow than
    relative to the running max). A reference only moves when the running max crosses a whole nat, and every rescale
    factor is exp(m_old - m_new), exactly 1 otherwise. So a term is rescaled at most (span of the block's g) + 1 times,
    however long the block, instead of once per running-max record (which made the rounding of ex2.approx pile up
    linearly in the block length for keys whose g grows with rank, e.g. a constant other realizer or g = slope * pos).
    The carry across tiles is a compensated (TwoSum) sum for the same reason: Lb / T sequential additions of similar
    tile sums otherwise round the same way every time.
    """
    pid = tl.program_id(0)
    inner = BH * 2 * NDS
    task = pid // inner
    rem = pid - task * inner
    bh = rem // (2 * NDS)
    rem = rem - bh * (2 * NDS)
    direction = rem // NDS
    dsl = rem - direction * NDS
    # task -> (level, block), levels in descending order
    lev = 0
    k = 0
    nb = 1
    found = 0
    t_rem = task
    for li in range(0, NLEV):
        lv = NLEV - 1 - li
        nb_l = ((nc - (1 << lv) - 1) >> (lv + 1)) + 1  # >= 1 for every level < NLEV
        take = (found == 0) & (t_rem < nb_l)
        lev = tl.where(take, lv, lev)
        k = tl.where(take, t_rem, k)
        nb = tl.where(take, nb_l, nb)
        found = tl.where(take, 1, found)
        t_rem = tl.where(found == 1, t_rem, t_rem - nb_l)
    Lb = C << lev
    blk_row = _level_row(lev, BH, nc, NLEV, C) + (bh.to(tl.int64) * nb + k) * Lb  # first row of this block
    pos0 = k.to(tl.int64) * 2 * Lb
    seq = bh.to(tl.int64) * N
    cols = dsl * BD + tl.arange(0, BD)
    cmask = cols < D
    lane = tl.arange(0, T)
    is_last = lane == T - 1
    if direction == 0:
        S_ptr = S1_ptr
        mz_ptr = mz1_ptr
    else:
        S_ptr = S2_ptr
        mz_ptr = mz2_ptr

    # carry: the state of all slots scanned so far, relative to its (whole) reference m_c, as compensated sums
    # (S_c + Se_c, z_c + ze_c): without the compensation terms the rounding of the Lb / T sequential carry additions
    # piles up linearly in the block length when the tiles add similar sums (near-uniform weights).
    m_c = tl.full((), _NEG, tl.float32)
    z_c = tl.zeros((), tl.float32)
    ze_c = tl.zeros((), tl.float32)
    S_c = tl.zeros((BD,), tl.float32)
    Se_c = tl.zeros((BD,), tl.float32)
    for t in range(0, Lb // T):
        i = t * T + lane
        slot = tl.where(direction == 0, i, Lb - 1 - i)  # the suffix scan walks the block backwards
        row = blk_row + slot
        pos = pos0 + tl.load(perm_ptr + row)
        gk = tl.load(g_ptr + (seq + pos) * 2 + direction)
        valid = pos != 0  # the sink is scored by b, never by the tree
        gm = tl.where(valid, gk, _NEG)
        inc, exc = tl.associative_scan((gm, tl.full((T,), _NEG, tl.float32)), 0, _max_incl_excl)
        R = tl.ceil(tl.maximum(inc, m_c))  # reference of slot i: its running max rounded up to whole nats
        Rp = tl.ceil(tl.maximum(exc, m_c))  # reference of slot i - 1
        a = tl.exp(Rp - R)  # exactly 1 unless the reference moves at slot i
        w = tl.where(valid, tl.exp(gm - R), 0.0)
        v = tl.load(v_ptr + (seq + pos)[:, None] * D + cols[None, :], mask=cmask[None, :], other=0.0).to(tl.float32)
        _, B = tl.associative_scan((tl.broadcast_to(a[:, None], (T, BD)), w[:, None] * v), 0, _affine)
        _, Bz = tl.associative_scan((a, w), 0, _affine)
        cf = tl.exp(m_c - R)  # the carry rescaled to each slot's reference in one factor, not a product of a's
        B = B + cf[:, None] * Se_c[None, :]
        Bz = Bz + cf * ze_c
        S = cf[:, None] * S_c[None, :] + B
        z = cf * z_c + Bz
        tl.store(S_ptr + row[:, None] * D + cols[None, :], S, mask=cmask[None, :])
        if dsl == 0:
            tl.store(mz_ptr + row * 2, R)
            tl.store(mz_ptr + row * 2 + 1, z)
        # next carry = the last slot's state, (cf S_c) + B at the last lane, summed with its rounding error (TwoSum)
        R_l = tl.max(R, axis=0)
        cf_l = tl.exp(m_c - R_l)
        hi = cf_l * S_c
        lo = tl.sum(tl.where(is_last[:, None], B, 0.0), axis=0)
        S_c = hi + lo
        bb = S_c - hi
        Se_c = (hi - (S_c - bb)) + (lo - bb)
        hz = cf_l * z_c
        lz = tl.sum(tl.where(is_last, Bz, 0.0), axis=0)
        z_c = hz + lz
        bz = z_c - hz
        ze_c = (hz - (z_c - bz)) + (lz - bz)
        m_c = R_l


@triton.jit
def _search_kernel(f_ptr, rank_ptr, r_ptr, N, nc, NLEV,
                   C: tl.constexpr, LOG2C: tl.constexpr, BQ: tl.constexpr, NL: tl.constexpr):
    """r[bh, l, x] = #keys with rank <= a(x) in the level-l block read by query x (0 where bit l of x's chunk is 0).

    A branchless binary search (binary lifting) over a (BQ, NL) tile of (query, level) pairs: LOG2C + NLEV dependent
    loads, whose steps sum to 2^(LOG2C + NLEV) - 1 >= every block size.
    """
    qb = tl.program_id(0)
    bh = tl.program_id(1)
    seq = bh.to(tl.int64) * N
    q = qb * BQ + tl.arange(0, BQ)
    qmask = q < N
    c = q // C
    a = (tl.load(f_ptr + (seq + q) * 2, mask=qmask, other=0.0)
         - tl.load(f_ptr + (seq + q) * 2 + 1, mask=qmask, other=0.0))
    lv = tl.arange(0, NL)
    has = (((c[:, None] >> lv[None, :]) & 1) == 1) & (lv < NLEV)[None, :] & qmask[:, None]
    Lb = C << lv
    nb = ((nc - (1 << lv) - 1) >> (lv + 1)) + 1  # garbage (unused) for lv >= NLEV
    row0 = (_level_row(lv, tl.num_programs(1), nc, NLEV, C)[None, :]
            + (bh.to(tl.int64) * nb[None, :] + (c[:, None] >> (lv[None, :] + 1))) * Lb[None, :])
    r = tl.zeros((BQ, NL), tl.int32)
    step = 1 << (LOG2C + NLEV - 1)
    for _ in range(0, LOG2C + NLEV):
        nxt = r + step
        ok = has & (nxt <= Lb[None, :])
        probe = tl.load(rank_ptr + row0 + (nxt - 1), mask=ok, other=0.0)
        r = tl.where(ok & (probe <= a[:, None]), nxt, r)
        step = step // 2
    tl.store(r_ptr + (bh.to(tl.int64) * NLEV + lv[None, :]) * N + q[:, None], r,
             mask=qmask[:, None] & (lv < NLEV)[None, :])


@triton.jit
def _query_kernel(f_ptr, g_ptr, b_ptr, v_ptr, r_ptr, S1_ptr, mz1_ptr, S2_ptr, mz2_ptr,
                  out_ptr, lse_ptr, lerr_ptr, A1_ptr, U1_ptr, p0_ptr,
                  N, nc, NLEV,
                  C: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr):
    """Output, lse (and its rounding residual), A1, U1 and p0 of BQ queries (one chunk) for one d-slice: tree states,
    intra tile, sink."""
    qb = tl.program_id(0)
    bh = tl.program_id(1)
    dsl = tl.program_id(2)
    seq = bh.to(tl.int64) * N
    q = qb * BQ + tl.arange(0, BQ)
    qmask = q < N
    c = (qb * BQ) // C
    cols = dsl * BD + tl.arange(0, BD)
    cmask = cols < D
    f1 = tl.load(f_ptr + (seq + q) * 2, mask=qmask, other=0.0)
    f2 = tl.load(f_ptr + (seq + q) * 2 + 1, mask=qmask, other=0.0)
    a = f1 - f2

    # online-softmax accumulators, one per branch, sharing the running max m_acc
    m_acc = tl.full((BQ,), _NEG, tl.float32)
    z1 = tl.zeros((BQ,), tl.float32)
    z2 = tl.zeros((BQ,), tl.float32)
    acc1 = tl.zeros((BQ, BD), tl.float32)
    acc2 = tl.zeros((BQ, BD), tl.float32)

    # ---- inter: one dyadic block of whole chunks per set bit of c
    lrow = bh.to(tl.int64) * 0  # first buffer row of level li
    for li in range(0, NLEV):
        Lb = C << li
        nb = ((nc - (1 << li) - 1) >> (li + 1)) + 1
        if ((c >> li) & 1) == 1:
            base = lrow + (bh.to(tl.int64) * nb + (c >> (li + 1))) * Lb
            r = tl.load(r_ptr + (bh.to(tl.int64) * NLEV + li) * N + q, mask=qmask, other=0)
            ok1 = (r > 0) & qmask  # some key on branch 1: prefix state at slot r - 1
            ok2 = (r < Lb) & qmask  # some key on branch 2: suffix state at slot r
            i1 = base + tl.maximum(r - 1, 0)
            i2 = base + tl.minimum(r, Lb - 1)
            m1 = tl.load(mz1_ptr + i1 * 2, mask=ok1, other=0.0)
            w1 = tl.load(mz1_ptr + i1 * 2 + 1, mask=ok1, other=0.0)
            m2 = tl.load(mz2_ptr + i2 * 2, mask=ok2, other=0.0)
            w2 = tl.load(mz2_ptr + i2 * 2 + 1, mask=ok2, other=0.0)
            X1 = tl.load(S1_ptr + i1[:, None] * D + cols[None, :], mask=ok1[:, None] & cmask[None, :], other=0.0)
            X2 = tl.load(S2_ptr + i2[:, None] * D + cols[None, :], mask=ok2[:, None] & cmask[None, :], other=0.0)
            mm1 = tl.where(ok1, m1 - f1, _NEG)
            mm2 = tl.where(ok2, m2 - f2, _NEG)
            m_new = tl.maximum(m_acc, tl.maximum(mm1, mm2))
            sa = tl.exp(m_acc - m_new)
            s1 = tl.exp(mm1 - m_new)
            s2 = tl.exp(mm2 - m_new)
            acc1 = acc1 * sa[:, None] + X1 * s1[:, None]
            acc2 = acc2 * sa[:, None] + X2 * s2[:, None]
            z1 = z1 * sa + w1 * s1
            z2 = z2 * sa + w2 * s2
            m_acc = m_new
        lrow += (nb * Lb).to(tl.int64) * tl.num_programs(1)

    # ---- intra: keys c C .. x of the own chunk, dense. The row max over the whole chunk first (no exp, no V), then
    # P @ V over the key sub-tiles up to the block's last query only (causal skipping).
    kpos = c * C + tl.arange(0, C)
    kmask = kpos < N
    g1k = tl.load(g_ptr + (seq + kpos) * 2, mask=kmask, other=0.0)
    g2k = tl.load(g_ptr + (seq + kpos) * 2 + 1, mask=kmask, other=0.0)
    br1 = (g1k - g2k)[None, :] <= a[:, None]
    logit = tl.where(br1, g1k[None, :] - f1[:, None], g2k[None, :] - f2[:, None])
    vis = (kpos[None, :] <= q[:, None]) & (kpos[None, :] >= 1) & kmask[None, :]
    m_new = tl.maximum(m_acc, tl.max(tl.where(vis, logit, _NEG), axis=1))
    sa = tl.exp(m_acc - m_new)
    acc1 = acc1 * sa[:, None]
    acc2 = acc2 * sa[:, None]
    z1 = z1 * sa
    z2 = z2 * sa
    m_acc = m_new
    for k0 in range(c * C, qb * BQ + BQ, BK):
        kp = k0 + tl.arange(0, BK)
        kpm = kp < N
        g1s = tl.load(g_ptr + (seq + kp) * 2, mask=kpm, other=0.0)
        g2s = tl.load(g_ptr + (seq + kp) * 2 + 1, mask=kpm, other=0.0)
        b1 = (g1s - g2s)[None, :] <= a[:, None]
        lg = tl.where(b1, g1s[None, :] - f1[:, None], g2s[None, :] - f2[:, None])
        vs = (kp[None, :] <= q[:, None]) & (kp[None, :] >= 1) & kpm[None, :]
        P = tl.where(vs, tl.exp(lg - m_acc[:, None]), 0.0)
        P1 = tl.where(b1, P, 0.0)
        P2 = P - P1
        vt = tl.load(v_ptr + (seq + kp)[:, None] * D + cols[None, :], mask=kpm[:, None] & cmask[None, :],
                     other=0.0).to(tl.float32)
        acc1 += tl.dot(P1, vt, input_precision="ieee")
        acc2 += tl.dot(P2, vt, input_precision="ieee")
        z1 += tl.sum(P1, axis=1)
        z2 += tl.sum(P2, axis=1)

    # ---- sink: logit b(x), value v(0)
    bq = tl.load(b_ptr + seq + q, mask=qmask, other=0.0)
    m_new = tl.maximum(m_acc, bq)
    sa = tl.exp(m_acc - m_new)
    s0 = tl.exp(bq - m_new)
    z = (z1 + z2) * sa + s0
    inv = 1.0 / z
    v0 = tl.load(v_ptr + seq * D + cols, mask=cmask, other=0.0).to(tl.float32)
    u1 = acc1 * (sa * inv)[:, None]
    o = u1 + acc2 * (sa * inv)[:, None] + (s0 * inv)[:, None] * v0[None, :]
    optr = (seq + q)[:, None] * D + cols[None, :]
    omask = qmask[:, None] & cmask[None, :]
    tl.store(out_ptr + optr, o, mask=omask)
    tl.store(U1_ptr + optr, u1, mask=omask)
    if dsl == 0:
        # lse = m + log z rounded, and its exact rounding residual (TwoSum): the backward recomputes probabilities as
        # exp(logit - lse - lerr); from the rounded lse alone they would be off by up to |lse| 2^-24 (1e-5 at 200)
        lz = tl.log(z)
        lse = m_new + lz
        bb = lse - m_new
        tl.store(lse_ptr + seq + q, lse, mask=qmask)
        tl.store(lerr_ptr + seq + q, (m_new - (lse - bb)) + (lz - bb), mask=qmask)
        tl.store(A1_ptr + seq + q, z1 * sa * inv, mask=qmask)
        tl.store(p0_ptr + seq + q, s0 * inv, mask=qmask)


# ----------------------------------------------------------------------------------------------------------------------
# Backward kernels
# ----------------------------------------------------------------------------------------------------------------------

@triton.jit
def _bwd_prep_kernel(do_ptr, out_ptr, U1_ptr, v_ptr, A1_ptr, p0_ptr, dd_ptr, df_ptr, db_ptr, dv0_ptr,
                     w_ptr, dw_ptr, lerr_ptr, lerrw_ptr,
                     N, M, mi, D: tl.constexpr, BD: tl.constexpr, BQ: tl.constexpr, HAS_W: tl.constexpr):
    """Per query: D = <dO, out>, df1 = -(<dO, U1> - D A1), df2 = -(<dO, U2> - D A2), db = p0 (<dO, v(0)> - D); per
    program: sum_x p0(x) dO(x) over its BQ queries (a part of dV(0), summed afterwards).

    With HAS_W, the backward of component ``mi`` of a mixture, whose upstream gradient is w(x) dO(x) for the gate
    weights w = w_ptr[..., mi] of the (B, H, N, M) weights: df, db and the dV(0) parts are multiplied by w; dd stays
    D = <dO, out>, which is also dL/dw and is stored into dw_ptr[..., mi]; and lerrw = lerr - log w is written for the
    scan and intra kernels, which read it in place of the lse residual lerr, so that every probability they recompute,
    exp(logit - lse - lerrw), is w p (for w = 0: lerrw = +inf, so exactly 0).
    """
    qb = tl.program_id(0)
    bh = tl.program_id(1)
    seq = bh.to(tl.int64) * N
    q = qb * BQ + tl.arange(0, BQ)
    qmask = q < N
    A1 = tl.load(A1_ptr + seq + q, mask=qmask, other=0.0)
    p0 = tl.load(p0_ptr + seq + q, mask=qmask, other=0.0)
    if HAS_W:
        wq = tl.load(w_ptr + (seq + q) * M + mi, mask=qmask, other=0.0)
        p0w = p0 * wq
    else:
        p0w = p0
    prow = (bh.to(tl.int64) * tl.num_programs(0) + qb) * D
    dD = tl.zeros((BQ,), tl.float32)
    dU = tl.zeros((BQ,), tl.float32)
    dv0 = tl.zeros((BQ,), tl.float32)
    for d0 in range(0, D, BD):
        cols = d0 + tl.arange(0, BD)
        cmask = cols < D
        ptr = (seq + q)[:, None] * D + cols[None, :]
        m2 = qmask[:, None] & cmask[None, :]
        do = tl.load(do_ptr + ptr, mask=m2, other=0.0)
        dD += tl.sum(do * tl.load(out_ptr + ptr, mask=m2, other=0.0), axis=1)
        dU += tl.sum(do * tl.load(U1_ptr + ptr, mask=m2, other=0.0), axis=1)
        v0 = tl.load(v_ptr + seq * D + cols, mask=cmask, other=0.0).to(tl.float32)
        dv0 += tl.sum(do * v0[None, :], axis=1)
        tl.store(dv0_ptr + prow + cols, tl.sum(do * p0w[:, None], axis=0), mask=cmask)
    df1 = dD * A1 - dU
    df2 = dD * (1.0 - A1 - p0) - (dD - dU - p0 * dv0)  # A2 = 1 - A1 - p0, <dO, U2> = D - <dO, U1> - p0 <dO, v(0)>
    db = p0 * (dv0 - dD)
    if HAS_W:
        df1 = df1 * wq
        df2 = df2 * wq
        db = db * wq
        tl.store(dw_ptr + (seq + q) * M + mi, dD, mask=qmask)
        lerr = tl.load(lerr_ptr + seq + q, mask=qmask, other=0.0)
        tl.store(lerrw_ptr + seq + q, lerr - tl.log(wq), mask=qmask)
    tl.store(dd_ptr + seq + q, dD, mask=qmask)
    tl.store(df_ptr + (seq + q) * 2, df1, mask=qmask)
    tl.store(df_ptr + (seq + q) * 2 + 1, df2, mask=qmask)
    tl.store(db_ptr + seq + q, db, mask=qmask)


@triton.jit
def _bwd_scan_kernel(perm_ptr, f_ptr, lse_ptr, lerr_ptr, do_ptr, dd_ptr, S1_ptr, mz1_ptr, S2_ptr, mz2_ptr,
                     N, BH, nc, NLEV,
                     C: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, NDS: tl.constexpr, T: tl.constexpr):
    """Transposed ``_scan_kernel``: states of the rank-sorted QUERY blocks, read by the keys of the chunks before them.

    Block k of level l holds the queries of the odd level-l block, positions (2k + 1) Lb .. (2k + 2) Lb - 1
    (Lb = C 2^l), sorted by a(x); the keys of the even block before it read it. Its queries x carry the signed payload
    [dO(x), D(x)] with log-weight lw1(x) = -f1(x) - lse(x) on branch 1 (keys with r(y) <= a(x): the SUFFIX of the sorted
    block from the first a(x) >= r(y); direction 1, into S1 / mz1) and lw2(x) = -f2(x) - lse(x) on branch 2
    (r(y) > a(x): the PREFIX of the queries with a(x) < r(y); direction 0, into S2 / mz2). mz holds [reference m,
    sum exp(lw - m) D].

    The numerics are those of ``_scan_kernel`` (whole-nat references, one direct carry factor, compensated carry);
    signed payloads are fine, since nothing takes a log of S. The log-weight is a large number (|f| + |lse|) whose
    exponential is later multiplied by exp(g(y) + m) ~ p(x, y): rounded to fp32 it would shift p by |lw| 2^-24 relative,
    so it is carried as a hi / lo pair (TwoSum of -f and -lse, minus the forward's lse residual) and the exponent is
    (hi - m) + lo, exact up to the rounding of a small number. A block may run past N (the last one, when nc is not a
    power of two): its padding slots have rank +inf and sort last, and the scan stops after the last real query's tile.
    For a mixture component, lerr is the prep kernel's lerr - log w (w <= 1 the gate weight): the weights become
    w exp(lw), the payload that of the component's upstream gradient w dO, and the exponent is still <= 0.
    """
    pid = tl.program_id(0)
    inner = BH * 2 * NDS
    task = pid // inner
    rem = pid - task * inner
    bh = rem // (2 * NDS)
    rem = rem - bh * (2 * NDS)
    direction = rem // NDS
    dsl = rem - direction * NDS
    lev = 0
    k = 0
    nb = 1
    found = 0
    t_rem = task
    for li in range(0, NLEV):
        lv = NLEV - 1 - li
        nb_l = ((nc - (1 << lv) - 1) >> (lv + 1)) + 1
        take = (found == 0) & (t_rem < nb_l)
        lev = tl.where(take, lv, lev)
        k = tl.where(take, t_rem, k)
        nb = tl.where(take, nb_l, nb)
        found = tl.where(take, 1, found)
        t_rem = tl.where(found == 1, t_rem, t_rem - nb_l)
    Lb = C << lev
    blk_row = _level_row(lev, BH, nc, NLEV, C) + (bh.to(tl.int64) * nb + k) * Lb
    pos0 = (2 * k.to(tl.int64) + 1) * Lb  # first query position of the block
    nreal = tl.minimum(N - pos0, Lb)  # >= 1: the block starts in chunk 2^l (2k + 1) < nc
    ntiles = (nreal + T - 1) // T
    seq = bh.to(tl.int64) * N
    cols = dsl * BD + tl.arange(0, BD)
    cmask = cols < D
    lane = tl.arange(0, T)
    is_last = lane == T - 1
    if direction == 0:
        S_ptr = S2_ptr
        mz_ptr = mz2_ptr
    else:
        S_ptr = S1_ptr
        mz_ptr = mz1_ptr
    fcol = 1 - direction

    m_c = tl.full((), _NEG, tl.float32)
    z_c = tl.zeros((), tl.float32)
    ze_c = tl.zeros((), tl.float32)
    S_c = tl.zeros((BD,), tl.float32)
    Se_c = tl.zeros((BD,), tl.float32)
    for t in range(0, ntiles):
        i = t * T + lane
        slot = tl.where(direction == 0, i, ntiles * T - 1 - i)
        row = blk_row + slot
        pos = pos0 + tl.load(perm_ptr + row)
        valid = pos < N
        nf = -tl.load(f_ptr + (seq + pos) * 2 + fcol, mask=valid, other=0.0)
        nl = -tl.load(lse_ptr + seq + pos, mask=valid, other=0.0)
        hi = nf + nl  # lw = hi + lo exactly (TwoSum), minus the lse residual
        bv = hi - nf
        lo = ((nf - (hi - bv)) + (nl - bv)) - tl.load(lerr_ptr + seq + pos, mask=valid, other=0.0)
        gm = tl.where(valid, hi, _NEG)
        inc, exc = tl.associative_scan((gm, tl.full((T,), _NEG, tl.float32)), 0, _max_incl_excl)
        R = tl.ceil(tl.maximum(inc, m_c))
        Rp = tl.ceil(tl.maximum(exc, m_c))
        a = tl.exp(Rp - R)
        w = tl.where(valid, tl.exp((gm - R) + lo), 0.0)
        v = tl.load(do_ptr + (seq + pos)[:, None] * D + cols[None, :], mask=valid[:, None] & cmask[None, :],
                    other=0.0)
        wd = w * tl.load(dd_ptr + seq + pos, mask=valid, other=0.0)
        _, B = tl.associative_scan((tl.broadcast_to(a[:, None], (T, BD)), w[:, None] * v), 0, _affine)
        _, Bz = tl.associative_scan((a, wd), 0, _affine)
        cf = tl.exp(m_c - R)
        B = B + cf[:, None] * Se_c[None, :]
        Bz = Bz + cf * ze_c
        S = cf[:, None] * S_c[None, :] + B
        z = cf * z_c + Bz
        tl.store(S_ptr + row[:, None] * D + cols[None, :], S, mask=cmask[None, :])
        if dsl == 0:
            tl.store(mz_ptr + row * 2, R)
            tl.store(mz_ptr + row * 2 + 1, z)
        R_l = tl.max(R, axis=0)
        cf_l = tl.exp(m_c - R_l)
        hs = cf_l * S_c
        ls = tl.sum(tl.where(is_last[:, None], B, 0.0), axis=0)
        S_c = hs + ls
        bb = S_c - hs
        Se_c = (hs - (S_c - bb)) + (ls - bb)
        hz = cf_l * z_c
        lz = tl.sum(tl.where(is_last, Bz, 0.0), axis=0)
        z_c = hz + lz
        bz = z_c - hz
        ze_c = (hz - (z_c - bz)) + (lz - bz)
        m_c = R_l


@triton.jit
def _bwd_search_kernel(g_ptr, rank_ptr, s_ptr, N, nc, NLEV,
                       C: tl.constexpr, LOG2C: tl.constexpr, BQ: tl.constexpr, NL: tl.constexpr):
    """s[bh, l, y] = #queries with a(x) < r(y) in the level-l query block read by key y (0 where it reads none).

    Key y of chunk c reads level l iff bit l of c is 0 and the odd block after its even level-l block exists. The
    comparison is strict because the forward puts y on branch 1 of x iff r(y) <= a(x): slots >= s are y's branch-1
    queries, slots < s its branch-2 queries.
    """
    yb = tl.program_id(0)
    bh = tl.program_id(1)
    seq = bh.to(tl.int64) * N
    y = yb * BQ + tl.arange(0, BQ)
    ymask = y < N
    c = y // C
    rk = (tl.load(g_ptr + (seq + y) * 2, mask=ymask, other=0.0)
          - tl.load(g_ptr + (seq + y) * 2 + 1, mask=ymask, other=0.0))
    lv = tl.arange(0, NL)
    nb = ((nc - (1 << lv) - 1) >> (lv + 1)) + 1  # garbage (unused) for lv >= NLEV
    kq = c[:, None] >> (lv[None, :] + 1)
    has = ((((c[:, None] >> lv[None, :]) & 1) == 0) & (lv < NLEV)[None, :] & ymask[:, None]
           & (kq < nb[None, :]))
    Lb = C << lv
    row0 = (_level_row(lv, tl.num_programs(1), nc, NLEV, C)[None, :]
            + (bh.to(tl.int64) * nb[None, :] + kq) * Lb[None, :])
    s = tl.zeros((BQ, NL), tl.int32)
    step = 1 << (LOG2C + NLEV - 1)
    for _ in range(0, LOG2C + NLEV):
        nxt = s + step
        ok = has & (nxt <= Lb[None, :])
        probe = tl.load(rank_ptr + row0 + (nxt - 1), mask=ok, other=0.0)
        s = tl.where(ok & (probe < rk[:, None]), nxt, s)
        step = step // 2
    tl.store(s_ptr + (bh.to(tl.int64) * NLEV + lv[None, :]) * N + y[:, None], s,
             mask=ymask[:, None] & (lv < NLEV)[None, :])


@triton.jit
def _key_intra_kernel(f_ptr, g_ptr, lse_ptr, lerr_ptr, dd_ptr, do_ptr, v_ptr, dv_ptr, dg_ptr, N,
                      C: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr, BQ: tl.constexpr,
                      ACC: tl.constexpr):
    """dV (one d-slice) and dg1 / dg2 of BK keys of one chunk from the queries of the same chunk (dense tile); writes
    dV (adds to it with ACC) and writes dg.

    Flash-attention style in the transposed (keys x queries) orientation: P^T = exp(logit - lse) (branch by the rank
    rule), dP^T = V dO^T, dS^T = P^T (dP^T - D); dV = P^T dO, dg_k = row sums of dS^T over the branch-k queries. When
    one slice holds all of d (BD >= D) V^T is loaded once; otherwise every slice program recomputes dP^T over all of d
    in BD-wide steps and slice 0 writes dg. For a mixture component, lerr is the prep kernel's lerr - log w: P^T is
    then w P^T, which scales dS^T and P^T dO alike (the backward for the upstream gradient w dO).
    """
    kb = tl.program_id(0)
    bh = tl.program_id(1)
    dsl = tl.program_id(2)
    seq = bh.to(tl.int64) * N
    y = kb * BK + tl.arange(0, BK)
    ymask = y < N
    ykey = ymask & (y >= 1)  # key 0 is the sink: no order logit
    c = (kb * BK) // C
    cols = dsl * BD + tl.arange(0, BD)
    cmask = cols < D
    g1 = tl.load(g_ptr + (seq + y) * 2, mask=ymask, other=0.0)
    g2 = tl.load(g_ptr + (seq + y) * 2 + 1, mask=ymask, other=0.0)
    rk = g1 - g2
    vy = tl.load(v_ptr + (seq + y)[:, None] * D + cols[None, :], mask=ymask[:, None] & cmask[None, :],
                 other=0.0).to(tl.float32)
    acc = tl.zeros((BK, BD), tl.float32)
    dg1 = tl.zeros((BK,), tl.float32)
    dg2 = tl.zeros((BK,), tl.float32)
    q_end = tl.minimum((c + 1) * C, N)  # queries before kb BK cannot see these keys, those after the chunk use the tree
    for q0 in range((kb * BK // BQ) * BQ, q_end, BQ):
        x = q0 + tl.arange(0, BQ)
        xm = x < q_end
        f1 = tl.load(f_ptr + (seq + x) * 2, mask=xm, other=0.0)
        f2 = tl.load(f_ptr + (seq + x) * 2 + 1, mask=xm, other=0.0)
        lse = tl.load(lse_ptr + seq + x, mask=xm, other=0.0)
        lerr = tl.load(lerr_ptr + seq + x, mask=xm, other=0.0)
        dx = tl.load(dd_ptr + seq + x, mask=xm, other=0.0)
        br1 = rk[:, None] <= (f1 - f2)[None, :]  # (BK, BQ)
        lg = tl.where(br1, g1[:, None] - f1[None, :], g2[:, None] - f2[None, :])
        vis = (y[:, None] <= x[None, :]) & ykey[:, None] & xm[None, :]
        PT = tl.where(vis, tl.exp((lg - lse[None, :]) - lerr[None, :]), 0.0)
        dos = tl.load(do_ptr + (seq + x)[:, None] * D + cols[None, :], mask=xm[:, None] & cmask[None, :], other=0.0)
        if BD >= D:
            dPT = tl.dot(vy, tl.trans(dos), input_precision="ieee")
        else:
            dPT = tl.zeros((BK, BQ), tl.float32)
            for d0 in range(0, D, BD):
                dc = d0 + tl.arange(0, BD)
                vt = tl.load(v_ptr + (seq + y)[:, None] * D + dc[None, :], mask=ymask[:, None] & (dc < D)[None, :],
                             other=0.0).to(tl.float32)
                dot = tl.load(do_ptr + (seq + x)[:, None] * D + dc[None, :], mask=xm[:, None] & (dc < D)[None, :],
                              other=0.0)
                dPT += tl.dot(vt, tl.trans(dot), input_precision="ieee")
        dST = PT * (dPT - dx[None, :])
        dg1 += tl.sum(tl.where(br1, dST, 0.0), axis=1)
        dg2 += tl.sum(tl.where(br1, 0.0, dST), axis=1)
        acc += tl.dot(PT, dos, input_precision="ieee")
    if ACC:  # after the loop: loading the old dV into acc before it made ptxas spill (intra 1.27 -> 4.76 ms)
        acc += tl.load(dv_ptr + (seq + y)[:, None] * D + cols[None, :], mask=ymask[:, None] & cmask[None, :],
                       other=0.0)
    tl.store(dv_ptr + (seq + y)[:, None] * D + cols[None, :], acc, mask=ymask[:, None] & cmask[None, :])
    if dsl == 0:
        tl.store(dg_ptr + (seq + y) * 2, dg1, mask=ymask)
        tl.store(dg_ptr + (seq + y) * 2 + 1, dg2, mask=ymask)


@triton.jit
def _key_tree_kernel(g_ptr, v_ptr, s_ptr, S1_ptr, mz1_ptr, S2_ptr, mz2_ptr, dv_ptr, dg_ptr, N, nc, NLEV,
                     C: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
    """Adds the later chunks' share to dV (one d-slice) of BK keys of one chunk and writes their dg1 / dg2 parts.

    For every level l with bit l of the key's chunk clear, the odd sibling block's branch-1 suffix state at slot s and
    branch-2 prefix state at slot s - 1. Every term is a probability times an upstream quantity, so the sum is linear:
    a state (m, S, zD) of branch k contributes exp(g_k(y) + m) [S, <v(y), S> - zD], whose exponent is log p(x, y) of
    the state's largest query up to the whole-nat rounding of m (so <= ~1) and exact for the dominant terms (m is an
    integer). dg parts are per d-slice (the zD terms go to slice 0) and summed afterwards.
    """
    kb = tl.program_id(0)
    bh = tl.program_id(1)
    dsl = tl.program_id(2)
    BH = tl.num_programs(1)
    seq = bh.to(tl.int64) * N
    y = kb * BK + tl.arange(0, BK)
    ymask = y < N
    ykey = ymask & (y >= 1)
    c = (kb * BK) // C
    cols = dsl * BD + tl.arange(0, BD)
    cmask = cols < D
    g1 = tl.load(g_ptr + (seq + y) * 2, mask=ymask, other=0.0)
    g2 = tl.load(g_ptr + (seq + y) * 2 + 1, mask=ymask, other=0.0)
    optr = dv_ptr + (seq + y)[:, None] * D + cols[None, :]
    omask = ymask[:, None] & cmask[None, :]
    vy = tl.load(v_ptr + (seq + y)[:, None] * D + cols[None, :], mask=omask, other=0.0).to(tl.float32)
    acc = tl.load(optr, mask=omask, other=0.0)  # the intra share, written by _key_intra_kernel
    dg1 = tl.zeros((BK,), tl.float32)
    dg2 = tl.zeros((BK,), tl.float32)
    first = (dsl == 0).to(tl.float32)
    lrow = bh.to(tl.int64) * 0  # first buffer row of level li
    for li in range(0, NLEV):
        Lb = C << li
        nb = ((nc - (1 << li) - 1) >> (li + 1)) + 1
        kq = c >> (li + 1)
        if (((c >> li) & 1) == 0) & (kq < nb):
            nreal = tl.minimum(N - (2 * kq + 1) * Lb, Lb)
            base = lrow + (bh.to(tl.int64) * nb + kq) * Lb
            s = tl.load(s_ptr + (bh.to(tl.int64) * NLEV + li) * N + y, mask=ymask, other=0)
            ok1 = (s < nreal) & ykey  # some branch-1 query: suffix state at slot s
            ok2 = (s > 0) & ykey  # some branch-2 query: prefix state at slot s - 1
            i1 = base + tl.minimum(s, nreal - 1)
            i2 = base + tl.maximum(s - 1, 0)
            m1 = tl.load(mz1_ptr + i1 * 2, mask=ok1, other=0.0)
            w1 = tl.load(mz1_ptr + i1 * 2 + 1, mask=ok1, other=0.0)
            m2 = tl.load(mz2_ptr + i2 * 2, mask=ok2, other=0.0)
            w2 = tl.load(mz2_ptr + i2 * 2 + 1, mask=ok2, other=0.0)
            X1 = tl.load(S1_ptr + i1[:, None] * D + cols[None, :], mask=ok1[:, None] & cmask[None, :], other=0.0)
            X2 = tl.load(S2_ptr + i2[:, None] * D + cols[None, :], mask=ok2[:, None] & cmask[None, :], other=0.0)
            e1 = tl.where(ok1, tl.exp(g1 + m1), 0.0)
            e2 = tl.where(ok2, tl.exp(g2 + m2), 0.0)
            acc += e1[:, None] * X1 + e2[:, None] * X2
            dg1 += e1 * (tl.sum(vy * X1, axis=1) - first * w1)
            dg2 += e2 * (tl.sum(vy * X2, axis=1) - first * w2)
        lrow += (nb * Lb).to(tl.int64) * BH
    tl.store(optr, acc, mask=omask)
    gp = dg_ptr + (dsl.to(tl.int64) * BH * N + seq + y) * 2
    tl.store(gp, dg1, mask=ymask)
    tl.store(gp + 1, dg2, mask=ymask)


@triton.jit
def _mix_sum_kernel(o_ptr, w_ptr, out_ptr, R, M, m0, o_stride,
                    D: tl.constexpr, MC: tl.constexpr, ACC: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
    """Gate-weighted sum of mixture components: out[r] = (out[r] if ACC else 0) + sum_{m < MC} w[r, m0 + m] O[m, r]
    for BR rows r of the (B H N) rows and one column tile, with O (MC, B H N, d) (component stride ``o_stride``) and w
    (B H N, M), in the order m = 0 .. MC - 1 (so MC = M at once and M calls with MC = 1 add the same terms in the same
    order)."""
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    rmask = r < R
    msk = rmask[:, None] & (cols < D)[None, :]
    ptr = r[:, None] * D + cols[None, :]
    if ACC:
        acc = tl.load(out_ptr + ptr, mask=msk, other=0.0)
    else:
        acc = tl.zeros((BR, BD), tl.float32)
    o_m = o_ptr
    for m in tl.static_range(MC):
        w = tl.load(w_ptr + r * M + m0 + m, mask=rmask, other=0.0)
        acc += w[:, None] * tl.load(o_m + ptr, mask=msk, other=0.0)
        o_m += o_stride  # pointer arithmetic is 64-bit: no int32 overflow of m * o_stride
    tl.store(out_ptr + ptr, acc, mask=msk)


# ----------------------------------------------------------------------------------------------------------------------
# Python API
# ----------------------------------------------------------------------------------------------------------------------

# Tile sizes, tuned on a GH200 at N = 32768, H = 16, d = 128, chunk 64. Larger query tiles (BLOCK_Q 32 / 64, or the
# whole d in one program) made ptxas fall back to a few registers with heavy spilling (10-20x slower query kernel).
BLOCK_Q = 16  # queries per query-kernel program; also the key sub-tile of the intra P @ V
BLOCK_D = 64  # value columns per query program; d is split into ceil(d / BLOCK_D) slices
SCAN_D = 32  # value columns per scan program
SCAN_T = 64  # sorted keys per scan tile
# Scan warps by value dtype (scan kernel ms at N = 32768; both choices give the same output bits):
SCAN_WARPS = 2  # fp32 values: 1.75 with 2 warps, 1.84 with 1
SCAN_WARPS_LOWP = 1  # bf16 / fp16 values: 1.45 with 1 warp, 2.03 with 2 (ptxas, not bytes: SCAN_D = 64 is no better)
SEARCH_Q = 64  # queries per search program
# Backward tile sizes (GH200, N = 32768, H = 16, d = 128, chunk 64)
PREP_Q = 32  # queries per prep program (D, df, db and a part of the sink's value gradient)
BWD_SCAN_D = 16  # dO columns per backward scan program (scan ms: 1.71 with 16 / 1 warp, 1.94 with 32 / 1, 2.05 32 / 2)
BWD_SCAN_WARPS = 1
# intra kernel ms: 32 x 32 ieee 1.27 (tf32x3: 1.14; 32 x 16: 2.14; 16 x 32: 2.23; 64 x 32 tf32x3 8 warps: 1.50; 2 warps:
# 14.2, spilling); a fused intra + tree kernel (16 keys, the whole d) took 2.57 ms against 1.27 + 0.69 split
INTRA_K = 32  # keys per intra program (capped at the chunk); the whole d in one program, so V^T is loaded once
INTRA_Q = 32  # queries per intra sub-tile
# Up to INTRA_FULL_D one intra program holds all of d; above, d-slices of INTRA_D columns that each recompute dP over
# all of d (intra ms at N = 32768, d = 192 / 256: 4.5 / 4.5 with 128-column slices, 4.8 / 8.2 with 64, 31 / 10.7 with
# the whole d in one program, which spills)
INTRA_FULL_D = 128
INTRA_D = 128
TREE_K = 16  # keys per tree program (tree kernel ms: 0.69 with 8 warps, 0.82 with 4; 32 keys or d = 128 slower)
TREE_D = 64  # dV columns per tree program
TREE_WARPS = 8
MIX_R = 32  # rows per program of the mixture's weighted sum
MIX_D = 64  # columns per program of the mixture's weighted sum


def _level_blocks(nc):
    """Blocks per level: nb_l = #even level-l blocks with a query chunk after them, for l = 0 .. ceil(log2 nc) - 1."""
    out, lv = [], 0
    while (1 << lv) < nc:
        out.append((nc - (1 << lv) - 1) // (2 << lv) + 1)
        lv += 1
    return out


def _check(f, g, b, V, chunk):
    if f.dim() != 4 or f.shape[-1] != 2 or g.shape != f.shape:
        raise ValueError(f"expected f, g of shape (B, H, N, 2), got {tuple(f.shape)} and {tuple(g.shape)}")
    if b.shape != f.shape[:-1] or V.dim() != 4 or V.shape[:-1] != f.shape[:-1]:
        raise ValueError(f"expected b (B, H, N) and V (B, H, N, d), got {tuple(b.shape)} and {tuple(V.shape)}")
    if chunk < 16 or chunk & (chunk - 1):
        raise ValueError(f"chunk must be a power of two >= 16, got {chunk}")
    if f.shape[2] == 0:
        raise ValueError("empty sequence")
    if not f.is_cuda and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("order_attention_triton needs CUDA tensors (or TRITON_INTERPRET=1)")


def _build_levels(g, V, chunk, timer=None):
    """Sort every level block by rank and scan it into prefix / suffix states.

    Returns (rank, S1, mz1, S2, mz2, nlev). Level l occupies rows lvl[l] .. lvl[l] + BH nb_l Lb_l of every buffer,
    lvl[l] = BH sum_{l' < l} nb_l' Lb_l' (nb_l from ``_level_blocks``; the kernels compute lvl[l] themselves), ordered
    (bh, block, sorted slot): rank holds the sorted key ranks, S1 / S2 (rows of d) the value sums of the prefix / suffix
    states and mz1 / mz2 their [reference m, normaliser z] (m = ceil(running max), see ``_scan_kernel``).
    """
    B, H, N, _ = g.shape
    BH, d = B * H, V.shape[-1]
    nc = triton.cdiv(N, chunk)
    nbs = _level_blocks(nc)
    sizes = [BH * nb * (chunk << lv) for lv, nb in enumerate(nbs)]
    total = max(sum(sizes), 1)
    dev = g.device
    rank = torch.empty(total, device=dev, dtype=torch.float32)
    S1 = torch.empty(total, d, device=dev, dtype=torch.float32)
    S2 = torch.empty(total, d, device=dev, dtype=torch.float32)
    mz1 = torch.empty(total, 2, device=dev, dtype=torch.float32)
    mz2 = torch.empty(total, 2, device=dev, dtype=torch.float32)
    base = [sum(sizes[:lv]) for lv in range(len(nbs))]
    if not nbs:
        return rank, S1, mz1, S2, mz2, 0

    P = (1 << len(nbs)) * chunk  # positions covered by the top level's block pair
    r_all = (g[..., 0] - g[..., 1]).reshape(BH, N)
    if P > N:
        r_all = torch.nn.functional.pad(r_all, (0, P - N))
    perm = torch.empty(total, device=dev, dtype=torch.int64)
    if timer:
        timer("sort", True)
    for lv, nb in enumerate(nbs):
        Lb = chunk << lv
        src = r_all[:, :P].view(BH, P // (2 * Lb), 2, Lb)[:, :nb, 0, :]  # the even blocks of level lv
        sl = slice(base[lv], base[lv] + sizes[lv])
        torch.sort(src, dim=-1, out=(rank[sl].view(BH, nb, Lb), perm[sl].view(BH, nb, Lb)))
    if timer:
        timer("sort", False)
        timer("scan", True)
    bd = min(SCAN_D, max(16, triton.next_power_of_2(d)))
    nds = triton.cdiv(d, bd)
    warps = SCAN_WARPS if V.element_size() == 4 else SCAN_WARPS_LOWP
    _scan_kernel[(sum(nbs) * BH * 2 * nds,)](perm, g, V, S1, mz1, S2, mz2, N, BH, nc, len(nbs),
                                            C=chunk, D=d, BD=bd, NDS=nds, T=min(SCAN_T, chunk), num_warps=warps)
    if timer:
        timer("scan", False)
    return rank, S1, mz1, S2, mz2, len(nbs)


def _values(V):
    """V as the kernels read it: fp32, bf16 or fp16 (any other dtype converted to fp32), contiguous."""
    if V.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        V = V.float()
    return V.contiguous()


def _on_device(t):
    """Makes ``t``'s CUDA device the current one (Triton launches on the current device); no-op off CUDA."""
    return torch.cuda.device(t.device) if t.is_cuda else contextlib.nullcontext()


def _order_attention_fwd(f, g, b, V, chunk=64, timer=None):
    """Forward pass; returns (out, lse, A1, U1, p0, lerr), see the module docstring for the layout.

    No host synchronisation and no host-to-device copy (the kernels compute the level offsets themselves), and a
    captured graph references only its own buffers, so the forward is CUDA-graph capturable once its kernels are
    compiled (also on the first call with a new shape).
    ``timer(name, start)``, if given, is called around the phases "sort", "scan", "search" and "query" (benchmarks).
    """
    _check(f, g, b, V, chunk)
    with _on_device(f):
        return _fwd(f, g, b, V, chunk, timer)


def _fwd(f, g, b, V, chunk, timer, out=None):
    """``_order_attention_fwd`` on the current device, writing the output into ``out`` (contiguous fp32) if given."""
    B, H, N, _ = f.shape
    d = V.shape[-1]
    f = f.float().contiguous()
    g = g.float().contiguous()
    b = b.float().contiguous()
    V = _values(V)
    rank, S1, mz1, S2, mz2, nlev = _build_levels(g, V, chunk, timer)
    nc = triton.cdiv(N, chunk)
    r = torch.empty(B * H * max(nlev, 1) * N, device=f.device, dtype=torch.int32)
    if timer:
        timer("search", True)
    if nlev:
        _search_kernel[(triton.cdiv(N, SEARCH_Q), B * H)](f, rank, r, N, nc, nlev, C=chunk,
                                                         LOG2C=int(math.log2(chunk)), BQ=SEARCH_Q,
                                                         NL=max(2, triton.next_power_of_2(nlev)), num_warps=4)
    if timer:
        timer("search", False)
        timer("query", True)
    if out is None:
        out = torch.empty(B, H, N, d, device=f.device, dtype=torch.float32)
    U1 = torch.empty_like(out)
    lse = torch.empty(B, H, N, device=f.device, dtype=torch.float32)
    lerr = torch.empty_like(lse)
    A1 = torch.empty_like(lse)
    p0 = torch.empty_like(lse)
    bd = min(BLOCK_D, max(16, triton.next_power_of_2(d)))  # tl.dot needs every dim >= 16
    _query_kernel[(triton.cdiv(N, BLOCK_Q), B * H, triton.cdiv(d, bd))](
        f, g, b, V, r, S1, mz1, S2, mz2, out, lse, lerr, A1, U1, p0, N, nc, nlev,
        C=chunk, D=d, BD=bd, BQ=BLOCK_Q, BK=BLOCK_Q, num_warps=4)
    if timer:
        timer("query", False)
    return out, lse, A1, U1, p0, lerr


def _bwd_build_levels(f, lse, lerr, dO, Dd, chunk, timer=None):
    """Sort every odd level block of queries by rank a(x) and scan it into the backward's suffix / prefix states.

    Same row layout as ``_build_levels`` (level l, bh, block k, sorted slot; nb_l blocks of Lb_l = C 2^l slots), but
    block k of level l holds the queries of positions (2k + 1) Lb_l .. (2k + 2) Lb_l - 1; padding positions (>= N) get
    rank +inf and sort last. Returns (rank, S1, mz1, S2, mz2, nlev): S1 / mz1 the branch-1 SUFFIX states and S2 / mz2
    the branch-2 PREFIX states of the payload [dO, D], see ``_bwd_scan_kernel``.
    """
    B, H, N, _ = f.shape
    BH, d = B * H, dO.shape[-1]
    nc = triton.cdiv(N, chunk)
    nbs = _level_blocks(nc)
    sizes = [BH * nb * (chunk << lv) for lv, nb in enumerate(nbs)]
    total = max(sum(sizes), 1)
    dev = f.device
    rank = torch.empty(total, device=dev, dtype=torch.float32)
    S1 = torch.empty(total, d, device=dev, dtype=torch.float32)
    S2 = torch.empty(total, d, device=dev, dtype=torch.float32)
    mz1 = torch.empty(total, 2, device=dev, dtype=torch.float32)
    mz2 = torch.empty(total, 2, device=dev, dtype=torch.float32)
    base = [sum(sizes[:lv]) for lv in range(len(nbs))]
    if not nbs:
        return rank, S1, mz1, S2, mz2, 0

    P = (1 << len(nbs)) * chunk
    a_all = (f[..., 0] - f[..., 1]).reshape(BH, N)  # bitwise the a(x) of the forward kernels
    if P > N:
        a_all = torch.nn.functional.pad(a_all, (0, P - N), value=float("inf"))
    perm = torch.empty(total, device=dev, dtype=torch.int64)
    if timer:
        timer("sort", True)
    for lv, nb in enumerate(nbs):
        Lb = chunk << lv
        src = a_all[:, :P].view(BH, P // (2 * Lb), 2, Lb)[:, :nb, 1, :]  # the odd blocks of level lv
        sl = slice(base[lv], base[lv] + sizes[lv])
        torch.sort(src, dim=-1, out=(rank[sl].view(BH, nb, Lb), perm[sl].view(BH, nb, Lb)))
    if timer:
        timer("sort", False)
        timer("scan", True)
    bd = min(BWD_SCAN_D, max(16, triton.next_power_of_2(d)))
    nds = triton.cdiv(d, bd)
    _bwd_scan_kernel[(sum(nbs) * BH * 2 * nds,)](perm, f, lse, lerr, dO, Dd, S1, mz1, S2, mz2, N, BH, nc, len(nbs),
                                                C=chunk, D=d, BD=bd, NDS=nds, T=min(SCAN_T, chunk),
                                                num_warps=BWD_SCAN_WARPS)
    if timer:
        timer("scan", False)
    return rank, S1, mz1, S2, mz2, len(nbs)


def _order_attention_bwd(f, g, b, V, out, lse, A1, U1, p0, lerr, dO, chunk=64, timer=None):
    """Backward pass from the forward's saved tensors (``_order_attention_fwd``); returns fp32 (df, dg, db, dV).

    ``b`` is not read (db comes from the saved p0); it is taken to mirror the forward's arguments. No host
    synchronisation and no host-to-device copy. ``timer(name, start)``, if given, is called around the phases
    "prep", "sort", "scan", "search", "intra" and "tree" (benchmarks).
    """
    with _on_device(f):
        B, H, N, _ = f.shape
        d = V.shape[-1]
        dV = torch.empty(B, H, N, d, device=f.device, dtype=torch.float32)
        dv0 = torch.empty(B * H, triton.cdiv(N, PREP_Q), d, device=f.device, dtype=torch.float32)
        df, dg, db = _bwd_core(f.float().contiguous(), g.float().contiguous(), _values(V), out, lse, A1, U1, p0, lerr,
                               dO.float().contiguous(), chunk, dV, dv0, timer)
        dV[:, :, 0] = dv0.sum(1).view(B, H, d)  # key 0 is only ever the sink
        return df, dg, db, dV


def _bwd_core(f, g, V, out, lse, A1, U1, p0, lerr, dO, chunk, dV, dv0, timer, acc=False, w=None, mi=0, dw=None):
    """The backward on the current device (f, g, dO fp32 contiguous, V from ``_values``); returns (df, dg, db).

    Writes the order keys' value gradients into dV (rows y >= 1; adds to dV with ``acc``) and leaves row 0 unchanged
    (zero without ``acc``); writes the parts of the sink's value gradient sum_x p0(x) dO(x), one per PREP_Q queries,
    into ``dv0`` (B H, ceil(N / PREP_Q), d) for the caller to sum. With gate weights ``w`` (B, H, N, M) this is the
    backward of mixture component ``mi`` for the upstream gradient w[..., mi] dO, without forming it: the prep kernel
    multiplies df, db and dV(0) by the weight and hands the scan and intra kernels lerr - log w in place of lerr (see
    ``_bwd_prep_kernel``), and it writes dL/dw[..., mi] = <dO, out> into ``dw[..., mi]``.
    """
    B, H, N, _ = f.shape
    BH, d = B * H, V.shape[-1]
    dev = f.device
    nc = triton.cdiv(N, chunk)
    dpow = max(16, triton.next_power_of_2(d))  # tl.dot needs every dim >= 16
    has_w = w is not None

    # per query: D, df1, df2, db; per program: a part of the sink's value gradient sum_x p0(x) dO(x)
    if timer:
        timer("prep", True)
    Dd = torch.empty(B, H, N, device=dev, dtype=torch.float32)
    df = torch.empty(B, H, N, 2, device=dev, dtype=torch.float32)
    db = torch.empty(B, H, N, device=dev, dtype=torch.float32)
    lerr_k = torch.empty_like(lerr) if has_w else lerr  # the residual the scan and intra kernels read
    _bwd_prep_kernel[(triton.cdiv(N, PREP_Q), BH)](dO, out, U1, V, A1, p0, Dd, df, db, dv0,
                                                   w if has_w else Dd, dw if has_w else Dd, lerr, lerr_k,
                                                   N, w.shape[-1] if has_w else 1, mi, D=d, BD=min(64, dpow),
                                                   BQ=PREP_Q, HAS_W=has_w, num_warps=4)
    if timer:
        timer("prep", False)

    rank, S1, mz1, S2, mz2, nlev = _bwd_build_levels(f, lse, lerr_k, dO, Dd, chunk, timer)
    s = torch.empty(BH * max(nlev, 1) * N, device=dev, dtype=torch.int32)
    if timer:
        timer("search", True)
    if nlev:
        _bwd_search_kernel[(triton.cdiv(N, SEARCH_Q), BH)](g, rank, s, N, nc, nlev, C=chunk,
                                                          LOG2C=int(math.log2(chunk)), BQ=SEARCH_Q,
                                                          NL=max(2, triton.next_power_of_2(nlev)), num_warps=4)
    if timer:
        timer("search", False)
        timer("intra", True)
    bd = min(TREE_D, dpow)
    nds = triton.cdiv(d, bd) if nlev else 0
    dgp = torch.empty(1 + nds, B, H, N, 2, device=dev, dtype=torch.float32)  # [intra, tree d-slices...]
    ik = min(INTRA_K, chunk)
    ibd = dpow if dpow <= INTRA_FULL_D else INTRA_D
    _key_intra_kernel[(triton.cdiv(N, ik), BH, triton.cdiv(d, ibd))](
        f, g, lse, lerr_k, Dd, dO, V, dV, dgp[0], N, C=chunk, D=d, BD=ibd, BK=ik, BQ=min(INTRA_Q, chunk), ACC=acc,
        num_warps=4, num_stages=1)
    if timer:
        timer("intra", False)
        timer("tree", True)
    if nlev:
        _key_tree_kernel[(triton.cdiv(N, TREE_K), BH, nds)](g, V, s, S1, mz1, S2, mz2, dV, dgp[1:], N, nc, nlev,
                                                           C=chunk, D=d, BD=bd, BK=TREE_K, num_warps=TREE_WARPS)
    if timer:
        timer("tree", False)
    dg = dgp[0] if nds == 0 else dgp.sum(0)
    return df, dg, db


class _OrderAttention(torch.autograd.Function):
    """Autograd wrapper: the forward saves (out, lse, A1, U1, p0, lerr) next to the inputs, the backward reads them."""

    @staticmethod
    def forward(ctx, f, g, b, V, chunk):
        saved = _order_attention_fwd(f, g, b, V, chunk)
        ctx.save_for_backward(f, g, b, V, *saved)
        ctx.chunk = chunk
        return saved[0]

    @staticmethod
    @torch.autograd.function.once_differentiable  # the Triton backward is not differentiable itself
    def backward(ctx, gout):
        f, g, b, V, *saved = ctx.saved_tensors
        df, dg, db, dV = _order_attention_bwd(f, g, b, V, *saved, gout, ctx.chunk)
        return df.to(f.dtype), dg.to(g.dtype), db.to(b.dtype), dV.to(V.dtype), None


def order_attention_triton(f, g, b, V, chunk=64):
    """Causal softmax(order scores) @ V, equal to ``order_attention(f, g, b, V, causal=True)``.

    Args:
        f: (B, H, N, 2) query realizer values (f1, f2).
        g: (B, H, N, 2) key realizer values (g1, g2); position 0 is the sink key and is scored by ``b`` instead.
        b: (B, H, N) sink logit per query.
        V: (B, H, N, d) values, fp32 / bf16 / fp16 (accumulated in fp32).
        chunk: keys per chunk (a power of two >= 16): the dense intra-chunk tile and the leaves of the tree.

    Returns:
        (B, H, N, d) fp32 attention output, differentiable in f, g, b and V (Triton backward, same chunked tree).
    """
    return _OrderAttention.apply(f, g, b, V, chunk)


def _mix_check(gates, comps, V, chunk):
    if not comps:
        raise ValueError("a mixture needs at least one component")
    for comp in comps:
        if len(comp) != 3:
            raise ValueError(f"expected components (f, g, b), got a tuple of {len(comp)}")
        _check(*comp, V, chunk)
    want = (*V.shape[:-1], len(comps))
    if tuple(gates.shape) != want:
        raise ValueError(f"expected gates of shape (B, H, N, M) = {want}, got {tuple(gates.shape)}")


def _mix_fwd(gates, comps, V, chunk, keep):
    """Mixture forward on the current device; returns (out, w, O, saved).

    w = softmax(gates) (B, H, N, M) fp32. With ``keep`` every component writes its output into O[m] of one
    (M, B, H, N, d) buffer, saved[m] holds its (lse, A1, U1, p0, lerr), and one ``_mix_sum_kernel`` launch forms
    sum_m w_m O[m]. Without (no gradient needed), each component's output is added into ``out`` and dropped, so only
    one component's tensors are alive at a time (O and saved are None).
    """
    B, H, N, d = V.shape
    M = len(comps)
    R = B * H * N
    w = torch.softmax(gates.float(), dim=-1).contiguous()
    V = _values(V)
    out = torch.empty(B, H, N, d, device=V.device, dtype=torch.float32)
    grid = (triton.cdiv(R, MIX_R), triton.cdiv(d, MIX_D))
    if keep:
        O = torch.empty(M, B, H, N, d, device=V.device, dtype=torch.float32)
        saved = [_fwd(f, g, b, V, chunk, None, out=O[m])[1:] for m, (f, g, b) in enumerate(comps)]
        _mix_sum_kernel[grid](O, w, out, R, M, 0, R * d, D=d, MC=M, ACC=False, BR=MIX_R, BD=MIX_D, num_warps=4)
        return out, w, O, saved
    O = torch.empty(B, H, N, d, device=V.device, dtype=torch.float32)  # reused by every component
    for m, (f, g, b) in enumerate(comps):
        _fwd(f, g, b, V, chunk, None, out=O)
        _mix_sum_kernel[grid](O, w, out, R, M, m, R * d, D=d, MC=1, ACC=m > 0, BR=MIX_R, BD=MIX_D, num_warps=4)
    return out, w, None, None


class _MixtureAttention(torch.autograd.Function):
    """Autograd wrapper of the gated mixture (``mixture_attention_triton``), fused over its components.

    The forward saves the component outputs O (one (M, B, H, N, d) buffer), each component's (lse, A1, U1, p0, lerr)
    and w = softmax(gates). The backward passes the upstream gradient once: component m's backward runs on it with
    every per-query term weighted by w_m (the prep kernel multiplies df, db and dV(0) by w_m and hands the scan and
    intra kernels lerr - log w_m in place of lerr), which is its backward for the upstream gradient w_m dO without
    forming that tensor. All components add into one dV (the intra kernel with ACC, the tree kernel always adds), and
    dgates follows from dL/dw_m = <dO, O[m]> (written by the prep kernel) through the softmax Jacobian.
    """

    @staticmethod
    def forward(ctx, gates, V, chunk, *flat):
        comps = [flat[i:i + 3] for i in range(0, len(flat), 3)]
        with _on_device(V):
            out, w, O, saved = _mix_fwd(gates, comps, V, chunk, keep=True)
        ctx.save_for_backward(V, w, O, *flat, *[t for s in saved for t in s])
        ctx.chunk, ctx.M, ctx.gates_dtype = chunk, len(comps), gates.dtype
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable  # the Triton backward is not differentiable itself
    def backward(ctx, gout):
        M, chunk = ctx.M, ctx.chunk
        V, w, O, *rest = ctx.saved_tensors
        flat, saved = rest[:3 * M], rest[3 * M:]
        B, H, N, d = V.shape
        with _on_device(V):
            dev = V.device
            gout = gout.float().contiguous()
            Vk = _values(V)
            dV = torch.empty(B, H, N, d, device=dev, dtype=torch.float32)
            dv0 = torch.empty(M, B * H, triton.cdiv(N, PREP_Q), d, device=dev, dtype=torch.float32)
            dw = torch.empty(B, H, N, M, device=dev, dtype=torch.float32)
            grads = []
            for m in range(M):
                f, g, b = flat[3 * m:3 * m + 3]
                df, dg, db = _bwd_core(f.float().contiguous(), g.float().contiguous(), Vk, O[m],
                                       *saved[5 * m:5 * m + 5], gout, chunk, dV, dv0[m], None, acc=m > 0, w=w, mi=m,
                                       dw=dw)
                grads += [df.to(f.dtype), dg.to(g.dtype), db.to(b.dtype)]
            dV[:, :, 0] = dv0.sum((0, 2)).view(B, H, d)  # key 0 is only ever the sink
            dgates = w * (dw - (w * dw).sum(-1, keepdim=True))  # softmax backward
        return (dgates.to(ctx.gates_dtype), dV.to(V.dtype), None, *grads)


def mixture_attention_triton(gates, comps, V, chunk=64):
    """Query-gated mixture of causal order heads: sum_m softmax(gates)_m order_attention_triton(*comps[m], V).

    One fused autograd function (``_MixtureAttention``) rather than autograd over the per-component outputs; when no
    gradient is needed (``torch.no_grad``, or no input requires grad) the forward keeps only one component's tensors
    alive at a time.

    Args:
        gates: (B, H, N, M) mixture logits.
        comps: M tuples (f, g, b) as in ``order_attention_triton``.
        V: (B, H, N, d) values shared by the components.
        chunk: see ``order_attention_triton``.

    Returns:
        (B, H, N, d) fp32 mixture output, differentiable in the gates, every component and V.
    """
    comps = [tuple(c) for c in comps]
    _mix_check(gates, comps, V, chunk)
    flat = [t for c in comps for t in c]
    if torch.is_grad_enabled() and any(t.requires_grad for t in (gates, V, *flat)):
        return _MixtureAttention.apply(gates, V, chunk, *flat)
    with _on_device(V):
        return _mix_fwd(gates, comps, V, chunk, keep=False)[0]
