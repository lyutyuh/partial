"""Causal order attention (``learning/order_attention.py``) as Triton kernels: a dense intra-chunk tile plus a dyadic
tree over whole chunks, like FlashAttention's causal split.

Semantics
---------
Per (batch, head): query realizers f(x) = (f1, f2), key realizers g(y) = (g1, g2), sink logit b(x), values v(y), and

    logit(x, y) = min(g1(y) - f1(x), g2(y) - f2(x))   for 1 <= y <= x,      logit(x, 0) = b(x)   (key 0 is the sink).

With query rank a(x) = f1(x) - f2(x) and key rank r(y) = g1(y) - g2(y), y is on branch 1 of x iff r(y) <= a(x) (logit
g1(y) - f1(x)), else on branch 2 (logit g2(y) - f2(x)). The kernels decide every branch by this one fp32 comparison of
the two ranks (also inside the dense tile), so the forward and a backward always agree on it; at a near-tie the two
candidate logits differ only by rounding.

Layout: f, g (batch, nheads, seqlen, 2); b (batch, nheads, seqlen); v (batch, nheads, seqlen, headdim). Every kernel
runs b * nheads + h on grid axis 0 (folded with the query / key block, or the level block of the scans), so
batch * nheads may exceed 65535, CUDA's cap on grid axes 1 and 2.

Chunked split
-------------
Positions are cut into nchunks = ceil(seqlen / chunk) chunks of ``chunk`` keys. A query x in chunk c sees

* INTRA: the keys y <= x of its own chunk (key 0 excluded), scored densely; P @ V on ``tl.dot`` in fp32.
* INTER: every key of chunks 0 .. c-1, as one dyadic block per set bit l of c: the 2^l chunks starting at chunk
  (c >> (l + 1)) << (l + 1). Only even-indexed level-l blocks are ever read, and only those with a query chunk after
  them, so level l (l = 0 .. ceil(log2 nchunks) - 1) holds nb_l = (nchunks - 2^l - 1) // 2^(l+1) + 1 blocks of
  Lb_l = chunk 2^l keys (about seqlen / 2 keys per level). No block contains the last chunk, so the tree never sees
  padding positions.

Forward kernels (one forward = ceil(log2 nchunks) segmented ``torch.sort`` calls + 3 Triton launches)
-------------------------------------------------------------------------------------------------------
1. Sort: per level, ``torch.sort`` of the key ranks along the last dim of a (batch_heads, nb_l, Lb_l) view (a segmented
   sort; equivalent to sorting the composite keys (block << 32 | rank) of ``order_attention._sort_within_blocks``).
2. ``_fwd_scan_kernel`` (all levels in one launch, top level first): at every sorted slot i of a block, the inclusive
   PREFIX state of exp(g1) [v, 1] (slots <= i) and the inclusive SUFFIX state of exp(g2) [v, 1] (slots >= i). A state
   (m, S, z) is relative to its reference m = ceil(own running max), a whole number of nats: S = sum exp(g - m) v,
   z = sum exp(g - m). Within a tile of BLOCK_SCAN sorted keys this is the linear recurrence
   S_i = exp(m_{i-1} - m_i) S_{i-1} + exp(g_i - m_i) v_i, run by ``tl.associative_scan`` with an affine combine, plus
   a sequential carry across tiles rescaled to each slot by one factor exp(m_carry - m_i) and summed with a TwoSum
   compensation. Every exponent is <= 0, so nothing overflows; a term only underflows when it is below 2^-126 of its
   state's reference; and a rescale factor is exactly 1 unless the running max crosses a whole nat, so neither the
   rescale roundings nor the carry additions accumulate with the block length (relative to the running max itself,
   the rounding of the per-record factors grew linearly with it: lse / A1 errors of 9x the 1e-5 tolerance at
   seqlen = 2^20 for a constant realizer).
3. ``_fwd_search_kernel``: split(x, l) = #keys of x's level-l block with rank <= a(x), a branchless binary search over
   a (queries, levels) tile. (Deviation from a search fused into the query kernel, measured at seqlen = 32768,
   nheads = 16, headdim = 128: fused, the chain of dependent loads ran at the query kernel's low occupancy and cost
   1.85 ms; split, the search takes 0.07 ms and the query kernel 1.55 ms.)
4. ``_fwd_kernel``, one program per (BLOCK_M queries, b * nheads + h, headdim block): for each set bit l of the chunk,
   the prefix state at slot split - 1 shifted by -f1(x) and the suffix state at slot split shifted by -f2(x) are
   combined online into per-branch register accumulators; then the intra tile and the sink exp(b(x)) [v(0), 1].

Cost O(seqlen chunk headdim) intra + O(seqlen headdim log(seqlen / chunk)) inter. The level buffers are transient
(freed after the forward). Layout, from ``_fwd_build_levels``: one flat row space, level-major, rows ordered (level l,
bh, block k, sorted slot) and level l starting at row lvl[l] = batch_heads sum_{l' < l} nb_l' Lb_l' (computed inside
the kernels, so no offset table is copied or cached); ``rank`` (rows,) fp32 sorted key ranks, ``perm`` (rows,) int64
key offsets inside the block, ``S1`` / ``S2`` (rows, headdim) fp32 prefix / suffix value sums, ``mz1`` / ``mz2``
(rows, 2) fp32 [reference m, normaliser z]; plus ``split`` (batch_heads, nlevels, seqlen) int32 from the search. Per
level and head that is about seqlen / 2 rows of 2 (headdim + 2) + 3 words: 2.9 GB at seqlen = 32768, nheads = 16,
headdim = 128 (bf16 SDPA: 0.13 GB).

Measured on a GH200 (scripts/bench_order_attention_triton.py, batch = 1, nheads = 16, headdim = 128, fp32, chunk 64):
3.9 ms at seqlen = 32768 and 8.1 ms at seqlen = 65536 (bf16 values: 3.5 / 7.4 ms), against 8.5 / 34.4 ms for the
fastest bf16 causal SDPA forward, cuDNN (FlashAttention-2, the default dispatch: 15.5 / 61.4 ms); break-even with cuDNN
near seqlen = 16384 (1.90 vs 1.80 ms), and at seqlen = 8192 it is 1.9x slower (0.98 vs 0.52 ms).

Saved for the backward (all fp32, contiguous, returned by ``_order_attn_forward`` in this order)
---------------------------------------------------------------------------------------------
* ``out`` (batch, nheads, seqlen, headdim): the attention output.
* ``softmax_lse`` (batch, nheads, seqlen): lse = log sum_y exp(logit(x, y)) over the sink and the visible keys, so
  p(x, y) = exp(logit - lse).
* ``branch1_mass`` (batch, nheads, seqlen): A1 = sum_{y on branch 1} p(x, y) (intra and inter keys).
* ``branch1_out`` (batch, nheads, seqlen, headdim): U1 = sum_{y on branch 1} p(x, y) v(y) (intra and inter keys).
* ``sink_prob`` (batch, nheads, seqlen): p0 = p(x, 0) = exp(b(x) - lse(x)).
* ``lse_residual`` (batch, nheads, seqlen): lerr, the rounding residual of lse = m + log z (TwoSum), so that
  lse + lerr is m + log z exactly.

Branch 2 follows as A2 = 1 - A1 - p0, U2 = out - U1 - p0 v(0). Query 0 sees only the sink: out(0) = v(0), p0(0) = 1.

Extreme inputs
--------------
-1e30 (``NEG``) is the kernels' finite stand-in for -inf, and finite sink logits below it count as NEG: with
b = finfo(float32).min (a masked sink) query 0 still gets out(0) = v(0), p0(0) = 1 (its saved lse is NEG instead of
b(0)); another row is unchanged unless all its logits are <= NEG, which is not supported (it then gets about v(0)).
b(0) = -inf leaves query 0 no finite logit and gives NaN, as in the torch references. A query masked by f_k = +inf (all
its branch-k logits -inf) is supported in the forward and the backward. NaN inputs propagate. The backward is first
order only: it raises under ``create_graph=True``.

Backward (``_order_attn_backward``)
-----------------------------------
With dO the upstream gradient, D(x) = <dO(x), out(x)> and ds(x, y) = p(x, y) (<dO(x), v(y)> - D(x)) (the softmax
backward):

* Per query, elementwise from the saved tensors (``_bwd_preprocess_kernel``): df1 = -(<dO, U1> - D A1),
  df2 = -(<dO, U2> - D A2), db = p0 (<dO, v(0)> - D); and dV(0) = sum_x p0(x) dO(x), as key 0 is only ever the sink.
* Per key y >= 1: dV(y) = e^{g1(y)} T1(y) + e^{g2(y)} T2(y) and dg_k(y) = e^{g_k(y)} (<v(y), T_k(y)> - T_kD(y)), with
  [T_k, T_kD](y) = sum of e^{-f_k(x) - lse(x)} [dO(x), D(x)] over the queries x >= y that put y on branch k (branch 1:
  a(x) >= r(y), branch 2: a(x) < r(y), the forward's tie rule). That is the transposed problem, solved by the same
  chunked split with queries and keys swapped:

  1. Sort: the queries of every ODD level block by a(x), the blocks read by the keys of the even block before them
     (``_bwd_build_levels``; same row layout, blocks of the last level may run past seqlen: padding sorts last).
  2. ``_bwd_scan_kernel``: the branch-1 SUFFIX and branch-2 PREFIX states of [dO, D] (signed) with log-weights
     -f1 - lse and -f2 - lse, numerically as ``_fwd_scan_kernel``.
  3. ``_bwd_search_kernel``: split(y, l) = #queries with a(x) < r(y) in the level-l block read by key y.
  4. ``_bwd_intra_kernel``: the dense tile of the key's own chunk, FlashAttention style in the keys x queries
     orientation (P^T = exp(logit - lse), dP^T = V dO^T, dS^T = P^T (dP^T - D); dV = P^T dO, dg_k = branch-k row sums).
  5. ``_bwd_tree_kernel``: adds exp(g_k(y) + m) [S, <v(y), S> - zD] of the state at slot split (branch 1) and
     split - 1 (branch 2) of every level. A linear sum suffices: every term is a probability times an upstream
     quantity, and the exponent is log p(x, y) of the state's largest query up to the whole-nat rounding of m.

  Probabilities are recomputed from lse, and the fp32 lse rounds by |lse| 2^-24: p = exp(logit - lse) alone was off by
  up to 1e-5 relative at |lse| ~ 200 (scale-50 integer inputs, whose fp32 logits are exact: dV errors of 3e-6 of the
  gradient scale, 4x the fp32 dense reference). So the forward also saves lse's TwoSum residual, the intra tile uses
  exp((logit - lse) - lerr) and the scan carries -f - lse as a hi / lo pair; after that dV is at 1.4e-7 of the scale.
  The tree's dg forms <v(y), sum_x p dO(x)> - sum_x p D(x) after the sum over queries (the dense softmax backward
  cancels per pair), which leaves dg errors up to ~1.5e-6 of the gradient scale at large magnitudes.

Backward cost: like the forward, O(seqlen chunk headdim) + O(seqlen headdim log(seqlen / chunk)) time and the same
transient level buffers. Measured on a GH200 (same setting as above): backward kernels 4.45 ms at seqlen = 32768
(prep 0.24, sorts 0.29, scan 1.77, search 0.07, intra 1.28, tree 0.70, glue 0.10); forward + backward 8.5 / 17.6 ms at
seqlen = 32768 / 65536 against 33.9 / 132.4 ms for cuDNN bf16 SDPA (4.0x / 7.5x), 2.0x at seqlen = 16384 (4.1 vs
8.3 ms) and about even at seqlen = 8192 (2.1-2.9 ms over runs, mostly host launch overhead, vs 2.0 ms).

Mixture (``mixture_attn_func``)
-------------------------------
out = sum_m w_m(x) out_m(x) with w = softmax(gates) over M components sharing v, as one autograd function
(``MixtureAttnFunc``) rather than autograd over the per-component outputs. The forward writes every component's
output into one (M, batch, nheads, seqlen, headdim) buffer and forms the weighted sum in one pass
(``_mixture_sum_kernel``). Component m's upstream gradient is w_m dO; instead of forming it, its backward takes dO
itself: the preprocess kernel (``IS_MIXTURE``) multiplies df, db and the dV(0) parts by w_m, writes dL/dw_m =
<dO, out_m>, and hands the scan and intra kernels lerr - log w_m in place of the lse residual lerr, so every
probability they recompute is w_m p (exactly 0 for w_m = 0, where that is +inf). That costs those two kernels no
extra load: multiplying by a gathered w_m in the scan and the intra tile instead was 0.45 ms slower at seqlen = 32768.
The fold rounds by about |log w_m| 2^-23, a relative error of that size in component m's probabilities, which are
scaled by w_m: at most max_w w |log w| 2^-23 ~ 5e-8 of the unweighted gradient scale. All components add into one dV
(the intra kernel with ``ACCUMULATE``, added after its loop), and dgates follows through the softmax Jacobian.
Without gradients (``torch.no_grad``, or no input requiring grad) each component's output is added into the result
and dropped, so only one component's tensors are alive at a time (eval memory as for M = 1 plus one output).
Measured (GH200, batch = 1, nheads = 16, headdim = 128, M = 4, chunk 64): forward + backward 8.4 / 34.2 / 72.0 ms at
seqlen = 8192 / 32768 / 65536 against 9.5 / 38.2 / 80.2 ms for autograd over the per-component outputs (whose
elementwise glue was 5.4 of its 37.4 ms of kernels at seqlen = 32768; now 1.1 ms including the weighted sum), i.e.
even with cuDNN bf16 SDPA at seqlen = 32768 (34.2 ms); peak memory 4.8 vs 5.0 GB there. No-grad forward 4.0 / 16.2 ms
at seqlen = 8192 / 32768 (autograd version 4.2 / 17.0 ms) at the same peak memory.
"""
import contextlib
import functools
import logging
import math
import os
from typing import Callable, Optional, Sequence

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Finite stand-in for -inf: (m_a - m_b) never becomes (-inf) - (-inf).
NEG = -1.0e30
_NEG = tl.constexpr(NEG)  # kernels may only read constexpr globals

Timer = Optional[Callable[[str, bool], None]]
"""``timer(phase, start)`` is called around the kernel phases of a forward / backward (benchmarks)."""

Component = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
"""One K = 2 order head: (f, g, b) realizers and sink logits."""


# ----------------------------------------------------------------------------------------------------------------------
# Tile sizes, tuned on a GH200 at seqlen = 32768, nheads = 16, headdim = 128, chunk 64
# ----------------------------------------------------------------------------------------------------------------------

# Forward. Larger query tiles (BLOCK_M 32 / 64, or the whole headdim in one program) made ptxas fall back to a few
# registers with heavy spilling (10-20x slower query kernel).
FWD_BLOCK_M = 16  # queries per program of _fwd_kernel; also the key sub-tile (BLOCK_N) of its intra P @ V
FWD_BLOCK_HEADDIM = 64  # value columns per program; headdim is split into ceil(headdim / 64) blocks
SCAN_BLOCK_HEADDIM = 32  # value columns per forward scan program
SCAN_BLOCK = 64  # sorted keys per scan tile (forward and backward scans)
# Scan warps by value dtype (scan kernel ms at seqlen = 32768; both choices give the same output bits):
SCAN_NUM_WARPS = 2  # fp32 values: 1.75 with 2 warps, 1.84 with 1
SCAN_NUM_WARPS_LOWP = 1  # bf16 / fp16 values: 1.45 with 1 warp, 2.03 with 2 (ptxas, not bytes: 64 columns no better)
SEARCH_BLOCK_M = 64  # queries (or keys) per search program
# Backward (GH200, seqlen = 32768, nheads = 16, headdim = 128, chunk 64)
BWD_PREPROCESS_BLOCK_M = 32  # queries per preprocess program (D, df, db and a part of the sink's value gradient)
# dO columns per backward scan program (scan ms: 1.71 with 16 / 1 warp, 1.94 with 32 / 1, 2.05 with 32 / 2)
BWD_SCAN_BLOCK_HEADDIM = 16
BWD_SCAN_NUM_WARPS = 1
# intra kernel ms: 32 x 32 ieee 1.27 (tf32x3: 1.14; 32 x 16: 2.14; 16 x 32: 2.23; 64 x 32 tf32x3 8 warps: 1.50; 2 warps:
# 14.2, spilling); a fused intra + tree kernel (16 keys, the whole headdim) took 2.57 ms against 1.27 + 0.69 split
# keys per intra program (capped at the chunk); the whole headdim in one program, so V^T is loaded once
BWD_INTRA_BLOCK_N = 32
BWD_INTRA_BLOCK_M = 32  # queries per intra sub-tile
# Up to BWD_INTRA_FULL_HEADDIM one intra program holds all of headdim; above, blocks of BWD_INTRA_BLOCK_HEADDIM columns
# that each recompute dP over all of headdim (intra ms at seqlen = 32768, headdim = 192 / 256: 4.5 / 4.5 with
# 128-column blocks, 4.8 / 8.2 with 64, 31 / 10.7 with the whole headdim in one program, which spills)
BWD_INTRA_FULL_HEADDIM = 128
BWD_INTRA_BLOCK_HEADDIM = 128
# keys per tree program (tree kernel ms: 0.69 with 8 warps, 0.82 with 4; 32 keys or 128 columns slower)
BWD_TREE_BLOCK_N = 16
BWD_TREE_BLOCK_HEADDIM = 64  # dV columns per tree program
BWD_TREE_NUM_WARPS = 8
MIX_BLOCK_M = 32  # rows per program of the mixture's weighted sum
MIX_BLOCK_HEADDIM = 64  # columns per program of the mixture's weighted sum


# ----------------------------------------------------------------------------------------------------------------------
# Kernel helpers
# ----------------------------------------------------------------------------------------------------------------------


@triton.jit
def _running_max_combine(incl_a, excl_a, incl_b, excl_b):
    # A segment is (max over it, max over it without its last element); combining keeps both.
    return tl.maximum(incl_a, incl_b), tl.maximum(incl_a, excl_b)


@triton.jit
def _affine_combine(a1, b1, a2, b2):
    # Composition of h -> a1 h + b1, then h -> a2 h + b2.
    return a1 * a2, b1 * a2 + b2


@triton.jit
def _two_sum(a, b):
    """Knuth's TwoSum: (s, err) with s = fl(a + b) and s + err = a + b exactly."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@triton.jit
def _num_blocks(nchunks, level):
    """nb_l: even level-``level`` blocks with a query chunk after them (>= 1 for every level below the top)."""
    return ((nchunks - (1 << level) - 1) >> (level + 1)) + 1


@triton.jit
def _level_row_offset(level, batch_heads, nchunks, nlevels, CHUNK: tl.constexpr):
    """First buffer row of level ``level`` (scalar or vector): batch_heads CHUNK sum_{l < level} nb_l 2^l, on device."""
    off = tl.zeros_like(level).to(tl.int64)
    for l in range(0, nlevels):
        off += tl.where(l < level, _num_blocks(nchunks, l) << l, 0).to(tl.int64)
    return off * batch_heads * CHUNK


@triton.jit
def _split_program_id(seqlen, BLOCK: tl.constexpr):
    """(block, off_bh) of a program whose grid axis 0 runs over bh cdiv(seqlen, BLOCK) + block (block fastest, the
    order of a (blocks, batch_heads) grid). batch_heads is folded into axis 0 because CUDA caps grid axes 1 and 2 at
    65535 programs."""
    pid = tl.program_id(0)
    nblocks = tl.cdiv(seqlen, BLOCK)
    off_bh = pid // nblocks
    return pid - off_bh * nblocks, off_bh


@triton.jit
def _scan_program_coords(batch_heads, nchunks, nlevels, NUM_HEADDIM_BLOCKS: tl.constexpr):
    """(level, block, nblocks, off_bh, direction, start_d) of one scan program.

    Program ids run over (task, bh, direction, headdim block) with the task slowest; tasks are the level blocks, top
    level first, so the long sequential scans of the big blocks start first and the many short ones fill the machine
    behind them.
    """
    pid = tl.program_id(0)
    inner = batch_heads * 2 * NUM_HEADDIM_BLOCKS
    task = pid // inner
    rem = pid - task * inner
    off_bh = rem // (2 * NUM_HEADDIM_BLOCKS)
    rem = rem - off_bh * (2 * NUM_HEADDIM_BLOCKS)
    direction = rem // NUM_HEADDIM_BLOCKS
    start_d = rem - direction * NUM_HEADDIM_BLOCKS
    # task -> (level, block), levels in descending order
    level = 0
    block = 0
    nblocks = 1
    found = 0
    t_rem = task
    for i in range(0, nlevels):
        l = nlevels - 1 - i
        nb_l = _num_blocks(nchunks, l)  # >= 1 for every level < nlevels
        take = (found == 0) & (t_rem < nb_l)
        level = tl.where(take, l, level)
        block = tl.where(take, t_rem, block)
        nblocks = tl.where(take, nb_l, nblocks)
        found = tl.where(take, 1, found)
        t_rem = tl.where(found == 1, t_rem, t_rem - nb_l)
    return level, block, nblocks, off_bh, direction, start_d


@triton.jit
def _running_references(gm, carry_m, BLOCK_SCAN: tl.constexpr):
    """References of a scan tile: (ref_m of slot i, ref_m of slot i - 1) = the running max through the slot, including
    the carry, rounded up to whole nats."""
    incl, excl = tl.associative_scan((gm, tl.full((BLOCK_SCAN,), _NEG, tl.float32)), 0, _running_max_combine)
    ref_m = tl.ceil(tl.maximum(incl, carry_m))
    ref_m_prev = tl.ceil(tl.maximum(excl, carry_m))
    return ref_m, ref_m_prev


@triton.jit
def _scan_tile(
    rescale,
    weight,
    payload,
    zweight,
    ref_m,
    carry_m,
    carry_S,
    carry_S_err,
    carry_z,
    carry_z_err,
    is_last,
    BLOCK_SCAN: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    """States (S, z) of one scan tile and the carry into the next tile.

    rescale[i] = exp(ref_m[i - 1] - ref_m[i]) and weight[i] = exp(logw[i] - ref_m[i]) give the tile's affine recurrence
    S_i = rescale_i S_{i-1} + weight_i payload_i (z_i likewise with zweight_i), run as an associative scan. The carry
    (the state of all slots scanned so far, relative to carry_m, as compensated sums carry_S + carry_S_err and
    carry_z + carry_z_err) is rescaled to each slot by one factor exp(carry_m - ref_m[i]), not a product of rescales.
    The new carry is the last slot's state, summed with its rounding error (TwoSum): without the compensation terms the
    rounding of the (block length / BLOCK_SCAN) sequential carry additions piles up linearly in the block length when
    the tiles add similar sums (near-uniform weights).
    """
    _, tile_S = tl.associative_scan(
        (tl.broadcast_to(rescale[:, None], (BLOCK_SCAN, BLOCK_HEADDIM)), weight[:, None] * payload), 0, _affine_combine
    )
    _, tile_z = tl.associative_scan((rescale, zweight), 0, _affine_combine)
    carry_scale = tl.exp(carry_m - ref_m)
    tile_S = tile_S + carry_scale[:, None] * carry_S_err[None, :]
    tile_z = tile_z + carry_scale * carry_z_err
    S = carry_scale[:, None] * carry_S[None, :] + tile_S
    z = carry_scale * carry_z + tile_z
    ref_m_last = tl.max(ref_m, axis=0)
    carry_scale_last = tl.exp(carry_m - ref_m_last)
    carry_S, carry_S_err = _two_sum(carry_scale_last * carry_S, tl.sum(tl.where(is_last[:, None], tile_S, 0.0), axis=0))
    carry_z, carry_z_err = _two_sum(carry_scale_last * carry_z, tl.sum(tl.where(is_last, tile_z, 0.0), axis=0))
    return S, z, ref_m_last, carry_S, carry_S_err, carry_z, carry_z_err


# ----------------------------------------------------------------------------------------------------------------------
# Forward kernels
# ----------------------------------------------------------------------------------------------------------------------


@triton.jit
def _fwd_scan_kernel(
    Perm,
    G,
    V,
    S1,
    MZ1,
    S2,
    MZ2,
    seqlen,
    batch_heads,
    nchunks,
    nlevels,
    CHUNK: tl.constexpr,
    HEADDIM: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    NUM_HEADDIM_BLOCKS: tl.constexpr,
    BLOCK_SCAN: tl.constexpr,
):
    """Prefix (direction 0) or suffix (direction 1) states of one rank-sorted level block, for one headdim block.

    One launch covers every level (``_scan_program_coords``). Perm holds, per sorted slot, the key's offset inside its
    block (block k of level l starts at position k 2^(l+1) CHUNK); the states go to the same rows as Perm:
    lvl[l] + (bh nb_l + k) Lb_l + slot.

    The state of slot i is kept relative to its reference m_i = ceil(running max through slot i), a whole number of
    nats: m_i >= every g in it (no exponent above 0) and m_i < running max + 1 (at most one nat more underflow than
    relative to the running max). A reference only moves when the running max crosses a whole nat, and every rescale
    factor is exp(m_old - m_new), exactly 1 otherwise. So a term is rescaled at most (span of the block's g) + 1 times,
    however long the block, instead of once per running-max record (which made the rounding of ex2.approx pile up
    linearly in the block length for keys whose g grows with rank, e.g. a constant other realizer or g = slope * pos).
    The carry across tiles is a compensated (TwoSum) sum for the same reason (``_scan_tile``).
    """
    level, block, nblocks, off_bh, direction, start_d = _scan_program_coords(
        batch_heads, nchunks, nlevels, NUM_HEADDIM_BLOCKS
    )
    block_len = CHUNK << level
    block_row = _level_row_offset(level, batch_heads, nchunks, nlevels, CHUNK) + (
        off_bh.to(tl.int64) * nblocks + block
    ) * block_len  # first row of this block
    pos0 = block.to(tl.int64) * 2 * block_len
    off_seq = off_bh.to(tl.int64) * seqlen
    offs_d = start_d * BLOCK_HEADDIM + tl.arange(0, BLOCK_HEADDIM)
    mask_d = offs_d < HEADDIM
    lane = tl.arange(0, BLOCK_SCAN)
    is_last = lane == BLOCK_SCAN - 1
    if direction == 0:
        S_out = S1
        MZ_out = MZ1
    else:
        S_out = S2
        MZ_out = MZ2

    # carry: the state of all slots scanned so far, relative to its (whole) reference carry_m, as compensated sums
    carry_m = tl.full((), _NEG, tl.float32)
    carry_z = tl.zeros((), tl.float32)
    carry_z_err = tl.zeros((), tl.float32)
    carry_S = tl.zeros((BLOCK_HEADDIM,), tl.float32)
    carry_S_err = tl.zeros((BLOCK_HEADDIM,), tl.float32)
    for t in range(0, block_len // BLOCK_SCAN):
        i = t * BLOCK_SCAN + lane
        slot = tl.where(direction == 0, i, block_len - 1 - i)  # the suffix scan walks the block backwards
        row = block_row + slot
        pos = pos0 + tl.load(Perm + row)
        g_k = tl.load(G + (off_seq + pos) * 2 + direction)
        valid = pos != 0  # the sink is scored by b, never by the tree
        gm = tl.where(valid, g_k, _NEG)
        ref_m, ref_m_prev = _running_references(gm, carry_m, BLOCK_SCAN)
        rescale = tl.exp(ref_m_prev - ref_m)  # exactly 1 unless the reference moves at slot i
        weight = tl.where(valid, tl.exp(gm - ref_m), 0.0)
        v = tl.load(V + (off_seq + pos)[:, None] * HEADDIM + offs_d[None, :], mask=mask_d[None, :], other=0.0).to(
            tl.float32
        )
        S, z, carry_m, carry_S, carry_S_err, carry_z, carry_z_err = _scan_tile(
            rescale,
            weight,
            v,
            weight,
            ref_m,
            carry_m,
            carry_S,
            carry_S_err,
            carry_z,
            carry_z_err,
            is_last,
            BLOCK_SCAN,
            BLOCK_HEADDIM,
        )
        tl.store(S_out + row[:, None] * HEADDIM + offs_d[None, :], S, mask=mask_d[None, :])
        if start_d == 0:
            tl.store(MZ_out + row * 2, ref_m)
            tl.store(MZ_out + row * 2 + 1, z)


@triton.jit
def _fwd_search_kernel(
    F,
    Rank,
    Split,
    seqlen,
    batch_heads,
    nchunks,
    nlevels,
    CHUNK: tl.constexpr,
    LOG2_CHUNK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_LEVELS: tl.constexpr,
):
    """Split[bh, l, x] = #keys with rank <= a(x) in the level-l block read by query x (0 where bit l of x's chunk is 0).

    A branchless binary search (binary lifting) over a (BLOCK_M, BLOCK_LEVELS) tile of (query, level) pairs:
    LOG2_CHUNK + nlevels dependent loads, whose steps sum to 2^(LOG2_CHUNK + nlevels) - 1 >= every block size.
    """
    start_m, off_bh = _split_program_id(seqlen, BLOCK_M)
    off_seq = off_bh.to(tl.int64) * seqlen
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seqlen
    chunk_idx = offs_m // CHUNK
    qrank = tl.load(F + (off_seq + offs_m) * 2, mask=mask_m, other=0.0) - tl.load(
        F + (off_seq + offs_m) * 2 + 1, mask=mask_m, other=0.0
    )
    levels = tl.arange(0, BLOCK_LEVELS)
    reads = (((chunk_idx[:, None] >> levels[None, :]) & 1) == 1) & (levels < nlevels)[None, :] & mask_m[:, None]
    block_len = CHUNK << levels
    nblocks = _num_blocks(nchunks, levels)  # garbage (unused) for levels >= nlevels
    block_row = _level_row_offset(levels, batch_heads, nchunks, nlevels, CHUNK)[None, :] + (
        off_bh.to(tl.int64) * nblocks[None, :] + (chunk_idx[:, None] >> (levels[None, :] + 1))
    ) * block_len[None, :]
    split = tl.zeros((BLOCK_M, BLOCK_LEVELS), tl.int32)
    step = 1 << (LOG2_CHUNK + nlevels - 1)
    for _ in range(0, LOG2_CHUNK + nlevels):
        probe_pos = split + step
        in_range = reads & (probe_pos <= block_len[None, :])
        probe = tl.load(Rank + block_row + (probe_pos - 1), mask=in_range, other=0.0)
        split = tl.where(in_range & (probe <= qrank[:, None]), probe_pos, split)
        step = step // 2
    tl.store(
        Split + (off_bh.to(tl.int64) * nlevels + levels[None, :]) * seqlen + offs_m[:, None],
        split,
        mask=mask_m[:, None] & (levels < nlevels)[None, :],
    )


@triton.jit
def _fwd_kernel(
    F,
    G,
    B,
    V,
    Split,
    S1,
    MZ1,
    S2,
    MZ2,
    Out,
    Lse,
    LseErr,
    A1,
    U1,
    P0,
    seqlen,
    batch_heads,
    nchunks,
    nlevels,
    CHUNK: tl.constexpr,
    HEADDIM: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Out, Lse (and its rounding residual LseErr), A1, U1 and P0 of BLOCK_M queries (one chunk) for one headdim block:
    tree states, intra tile, sink."""
    start_m, off_bh = _split_program_id(seqlen, BLOCK_M)
    start_d = tl.program_id(1)
    off_seq = off_bh.to(tl.int64) * seqlen
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seqlen
    chunk_idx = (start_m * BLOCK_M) // CHUNK
    offs_d = start_d * BLOCK_HEADDIM + tl.arange(0, BLOCK_HEADDIM)
    mask_d = offs_d < HEADDIM
    f1 = tl.load(F + (off_seq + offs_m) * 2, mask=mask_m, other=0.0)
    f2 = tl.load(F + (off_seq + offs_m) * 2 + 1, mask=mask_m, other=0.0)
    qrank = f1 - f2

    # online-softmax accumulators, one per branch, sharing the running max m_acc
    m_acc = tl.full((BLOCK_M,), _NEG, tl.float32)
    z1 = tl.zeros((BLOCK_M,), tl.float32)
    z2 = tl.zeros((BLOCK_M,), tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_HEADDIM), tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_HEADDIM), tl.float32)

    # ---- inter: one dyadic block of whole chunks per set bit of the chunk index
    level_row = off_bh.to(tl.int64) * 0  # first buffer row of the level (an int64 zero tensor, carried by the loop)
    for level in range(0, nlevels):
        block_len = CHUNK << level
        nblocks = _num_blocks(nchunks, level)
        if ((chunk_idx >> level) & 1) == 1:
            block_row = level_row + (off_bh.to(tl.int64) * nblocks + (chunk_idx >> (level + 1))) * block_len
            split = tl.load(Split + (off_bh.to(tl.int64) * nlevels + level) * seqlen + offs_m, mask=mask_m, other=0)
            has1 = (split > 0) & mask_m  # some key on branch 1: prefix state at slot split - 1
            has2 = (split < block_len) & mask_m  # some key on branch 2: suffix state at slot split
            row1 = block_row + tl.maximum(split - 1, 0)
            row2 = block_row + tl.minimum(split, block_len - 1)
            state_m1 = tl.load(MZ1 + row1 * 2, mask=has1, other=0.0)
            state_z1 = tl.load(MZ1 + row1 * 2 + 1, mask=has1, other=0.0)
            state_m2 = tl.load(MZ2 + row2 * 2, mask=has2, other=0.0)
            state_z2 = tl.load(MZ2 + row2 * 2 + 1, mask=has2, other=0.0)
            state_S1 = tl.load(
                S1 + row1[:, None] * HEADDIM + offs_d[None, :], mask=has1[:, None] & mask_d[None, :], other=0.0
            )
            state_S2 = tl.load(
                S2 + row2[:, None] * HEADDIM + offs_d[None, :], mask=has2[:, None] & mask_d[None, :], other=0.0
            )
            logit1 = tl.where(has1, state_m1 - f1, _NEG)
            logit2 = tl.where(has2, state_m2 - f2, _NEG)
            m_new = tl.maximum(m_acc, tl.maximum(logit1, logit2))
            alpha = tl.exp(m_acc - m_new)
            p1 = tl.exp(logit1 - m_new)
            p2 = tl.exp(logit2 - m_new)
            acc1 = acc1 * alpha[:, None] + state_S1 * p1[:, None]
            acc2 = acc2 * alpha[:, None] + state_S2 * p2[:, None]
            z1 = z1 * alpha + state_z1 * p1
            z2 = z2 * alpha + state_z2 * p2
            m_acc = m_new
        level_row += (nblocks * block_len).to(tl.int64) * batch_heads

    # ---- intra: keys chunk_idx CHUNK .. x of the own chunk, dense. The row max over the whole chunk first (no exp, no
    # V), then P @ V over the key sub-tiles up to the block's last query only (causal skipping).
    offs_chunk = chunk_idx * CHUNK + tl.arange(0, CHUNK)
    mask_chunk = offs_chunk < seqlen
    g1_chunk = tl.load(G + (off_seq + offs_chunk) * 2, mask=mask_chunk, other=0.0)
    g2_chunk = tl.load(G + (off_seq + offs_chunk) * 2 + 1, mask=mask_chunk, other=0.0)
    branch1_chunk = (g1_chunk - g2_chunk)[None, :] <= qrank[:, None]
    logit_chunk = tl.where(branch1_chunk, g1_chunk[None, :] - f1[:, None], g2_chunk[None, :] - f2[:, None])
    visible_chunk = (offs_chunk[None, :] <= offs_m[:, None]) & (offs_chunk[None, :] >= 1) & mask_chunk[None, :]
    m_new = tl.maximum(m_acc, tl.max(tl.where(visible_chunk, logit_chunk, _NEG), axis=1))
    alpha = tl.exp(m_acc - m_new)
    acc1 = acc1 * alpha[:, None]
    acc2 = acc2 * alpha[:, None]
    z1 = z1 * alpha
    z2 = z2 * alpha
    m_acc = m_new
    for start_n in range(chunk_idx * CHUNK, start_m * BLOCK_M + BLOCK_M, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen
        g1_n = tl.load(G + (off_seq + offs_n) * 2, mask=mask_n, other=0.0)
        g2_n = tl.load(G + (off_seq + offs_n) * 2 + 1, mask=mask_n, other=0.0)
        branch1 = (g1_n - g2_n)[None, :] <= qrank[:, None]
        logit = tl.where(branch1, g1_n[None, :] - f1[:, None], g2_n[None, :] - f2[:, None])
        visible = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] >= 1) & mask_n[None, :]
        p = tl.where(visible, tl.exp(logit - m_acc[:, None]), 0.0)
        p1 = tl.where(branch1, p, 0.0)
        p2 = p - p1
        v_n = tl.load(
            V + (off_seq + offs_n)[:, None] * HEADDIM + offs_d[None, :],
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        acc1 += tl.dot(p1, v_n, input_precision="ieee")
        acc2 += tl.dot(p2, v_n, input_precision="ieee")
        z1 += tl.sum(p1, axis=1)
        z2 += tl.sum(p2, axis=1)

    # ---- sink: logit b(x), value v(0). A finite b below NEG (e.g. finfo.min) is raised to NEG: query 0, which has
    # no keys, got m_new = NEG, s0 = exp(b - NEG) = 0, z = 0 and a NaN output; raised, it gets s0 = 1, out = v(0),
    # p0 = 1, as the torch references. Other rows are unchanged unless all their logits are <= NEG (exp(NEG - m) is
    # exactly 0 for m > NEG + 104). -inf and NaN are kept: query 0 then gives NaN, as the references.
    b_m = tl.load(B + off_seq + offs_m, mask=mask_m, other=0.0)
    b_m = tl.where((b_m < _NEG) & (b_m > float("-inf")), _NEG, b_m)
    m_new = tl.maximum(m_acc, b_m)
    alpha = tl.exp(m_acc - m_new)
    s0 = tl.exp(b_m - m_new)
    z = (z1 + z2) * alpha + s0
    inv_z = 1.0 / z
    v0 = tl.load(V + off_seq * HEADDIM + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    u1 = acc1 * (alpha * inv_z)[:, None]
    out = u1 + acc2 * (alpha * inv_z)[:, None] + (s0 * inv_z)[:, None] * v0[None, :]
    out_ptrs = (off_seq + offs_m)[:, None] * HEADDIM + offs_d[None, :]
    mask_md = mask_m[:, None] & mask_d[None, :]
    tl.store(Out + out_ptrs, out, mask=mask_md)
    tl.store(U1 + out_ptrs, u1, mask=mask_md)
    if start_d == 0:
        # lse = m + log z rounded, and its exact rounding residual (TwoSum): the backward recomputes probabilities as
        # exp(logit - lse - lerr); from the rounded lse alone they would be off by up to |lse| 2^-24 (1e-5 at 200)
        log_z = tl.log(z)
        lse, lse_err = _two_sum(m_new, log_z)
        tl.store(Lse + off_seq + offs_m, lse, mask=mask_m)
        tl.store(LseErr + off_seq + offs_m, lse_err, mask=mask_m)
        tl.store(A1 + off_seq + offs_m, z1 * alpha * inv_z, mask=mask_m)
        tl.store(P0 + off_seq + offs_m, s0 * inv_z, mask=mask_m)


# ----------------------------------------------------------------------------------------------------------------------
# Backward kernels
# ----------------------------------------------------------------------------------------------------------------------


@triton.jit
def _bwd_preprocess_kernel(
    DO,
    Out,
    U1,
    V,
    A1,
    P0,
    Delta,
    DF,
    DB,
    DVSink,
    W,
    DW,
    LseErr,
    LseErrW,
    seqlen,
    ncomponents,
    component,
    HEADDIM: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    IS_MIXTURE: tl.constexpr,
):
    """Per query: D = <dO, out>, df1 = -(<dO, U1> - D A1), df2 = -(<dO, U2> - D A2), db = p0 (<dO, v(0)> - D); per
    program: sum_x p0(x) dO(x) over its BLOCK_M queries (a part of dV(0), summed afterwards).

    With IS_MIXTURE, the backward of component ``component`` of a mixture, whose upstream gradient is w(x) dO(x) for
    the gate weights w = W[..., component] of the (batch, nheads, seqlen, ncomponents) weights: df, db and the dV(0)
    parts are multiplied by w; Delta stays D = <dO, out>, which is also dL/dw and is stored into DW[..., component];
    and LseErrW = lerr - log w is written for the scan and intra kernels, which read it in place of the lse residual
    lerr, so that every probability they recompute, exp(logit - lse - lerrw), is w p (for w = 0: lerrw = +inf, so
    exactly 0).
    """
    start_m, off_bh = _split_program_id(seqlen, BLOCK_M)
    off_seq = off_bh.to(tl.int64) * seqlen
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seqlen
    a1_m = tl.load(A1 + off_seq + offs_m, mask=mask_m, other=0.0)
    p0_m = tl.load(P0 + off_seq + offs_m, mask=mask_m, other=0.0)
    if IS_MIXTURE:
        w_m = tl.load(W + (off_seq + offs_m) * ncomponents + component, mask=mask_m, other=0.0)
        p0_w = p0_m * w_m
    else:
        p0_w = p0_m
    part_row = tl.program_id(0).to(tl.int64) * HEADDIM  # row bh cdiv(seqlen, BLOCK_M) + start_m of the DVSink parts
    delta = tl.zeros((BLOCK_M,), tl.float32)
    do_u1 = tl.zeros((BLOCK_M,), tl.float32)
    do_v0 = tl.zeros((BLOCK_M,), tl.float32)
    for start_d in range(0, HEADDIM, BLOCK_HEADDIM):
        offs_d = start_d + tl.arange(0, BLOCK_HEADDIM)
        mask_d = offs_d < HEADDIM
        ptrs = (off_seq + offs_m)[:, None] * HEADDIM + offs_d[None, :]
        mask_md = mask_m[:, None] & mask_d[None, :]
        do_m = tl.load(DO + ptrs, mask=mask_md, other=0.0)
        delta += tl.sum(do_m * tl.load(Out + ptrs, mask=mask_md, other=0.0), axis=1)
        do_u1 += tl.sum(do_m * tl.load(U1 + ptrs, mask=mask_md, other=0.0), axis=1)
        v0 = tl.load(V + off_seq * HEADDIM + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        do_v0 += tl.sum(do_m * v0[None, :], axis=1)
        tl.store(DVSink + part_row + offs_d, tl.sum(do_m * p0_w[:, None], axis=0), mask=mask_d)
    df1 = delta * a1_m - do_u1
    # A2 = 1 - A1 - p0, <dO, U2> = D - <dO, U1> - p0 <dO, v(0)>
    df2 = delta * (1.0 - a1_m - p0_m) - (delta - do_u1 - p0_m * do_v0)
    db = p0_m * (do_v0 - delta)
    if IS_MIXTURE:
        df1 = df1 * w_m
        df2 = df2 * w_m
        db = db * w_m
        tl.store(DW + (off_seq + offs_m) * ncomponents + component, delta, mask=mask_m)
        lse_err = tl.load(LseErr + off_seq + offs_m, mask=mask_m, other=0.0)
        tl.store(LseErrW + off_seq + offs_m, lse_err - tl.log(w_m), mask=mask_m)
    tl.store(Delta + off_seq + offs_m, delta, mask=mask_m)
    tl.store(DF + (off_seq + offs_m) * 2, df1, mask=mask_m)
    tl.store(DF + (off_seq + offs_m) * 2 + 1, df2, mask=mask_m)
    tl.store(DB + off_seq + offs_m, db, mask=mask_m)


@triton.jit
def _bwd_scan_kernel(
    Perm,
    F,
    Lse,
    LseErr,
    DO,
    Delta,
    S1,
    MZ1,
    S2,
    MZ2,
    seqlen,
    batch_heads,
    nchunks,
    nlevels,
    CHUNK: tl.constexpr,
    HEADDIM: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    NUM_HEADDIM_BLOCKS: tl.constexpr,
    BLOCK_SCAN: tl.constexpr,
):
    """Transposed ``_fwd_scan_kernel``: states of the rank-sorted QUERY blocks, read by the keys of the chunks before.

    Block k of level l holds the queries of the odd level-l block, positions (2k + 1) Lb .. (2k + 2) Lb - 1
    (Lb = CHUNK 2^l), sorted by a(x); the keys of the even block before it read it. Its queries x carry the signed
    payload [dO(x), D(x)] with log-weight lw1(x) = -f1(x) - lse(x) on branch 1 (keys with r(y) <= a(x): the SUFFIX of
    the sorted block from the first a(x) >= r(y); direction 1, into S1 / MZ1) and lw2(x) = -f2(x) - lse(x) on branch 2
    (r(y) > a(x): the PREFIX of the queries with a(x) < r(y); direction 0, into S2 / MZ2). MZ holds [reference m,
    sum exp(lw - m) D].

    The numerics are those of ``_fwd_scan_kernel`` (whole-nat references, one direct carry factor, compensated carry);
    signed payloads are fine, since nothing takes a log of S. The log-weight is a large number (|f| + |lse|) whose
    exponential is later multiplied by exp(g(y) + m) ~ p(x, y): rounded to fp32 it would shift p by |lw| 2^-24 relative,
    so it is carried as a hi / lo pair (TwoSum of -f and -lse, minus the forward's lse residual) and the exponent is
    (hi - m) + lo, exact up to the rounding of a small number. A block may run past seqlen (the last one, when nchunks
    is not a power of two): its padding slots have rank +inf and sort last, and the scan stops after the last real
    query's tile. For a mixture component, LseErr is the preprocess kernel's lerr - log w (w <= 1 the gate weight): the
    weights become w exp(lw), the payload that of the component's upstream gradient w dO, and the exponent is still
    <= 0.
    """
    level, block, nblocks, off_bh, direction, start_d = _scan_program_coords(
        batch_heads, nchunks, nlevels, NUM_HEADDIM_BLOCKS
    )
    block_len = CHUNK << level
    block_row = _level_row_offset(level, batch_heads, nchunks, nlevels, CHUNK) + (
        off_bh.to(tl.int64) * nblocks + block
    ) * block_len
    pos0 = (2 * block.to(tl.int64) + 1) * block_len  # first query position of the block
    nreal = tl.minimum(seqlen - pos0, block_len)  # >= 1: the block starts in chunk 2^l (2k + 1) < nchunks
    ntiles = (nreal + BLOCK_SCAN - 1) // BLOCK_SCAN
    off_seq = off_bh.to(tl.int64) * seqlen
    offs_d = start_d * BLOCK_HEADDIM + tl.arange(0, BLOCK_HEADDIM)
    mask_d = offs_d < HEADDIM
    lane = tl.arange(0, BLOCK_SCAN)
    is_last = lane == BLOCK_SCAN - 1
    if direction == 0:
        S_out = S2
        MZ_out = MZ2
    else:
        S_out = S1
        MZ_out = MZ1
    f_col = 1 - direction

    carry_m = tl.full((), _NEG, tl.float32)
    carry_z = tl.zeros((), tl.float32)
    carry_z_err = tl.zeros((), tl.float32)
    carry_S = tl.zeros((BLOCK_HEADDIM,), tl.float32)
    carry_S_err = tl.zeros((BLOCK_HEADDIM,), tl.float32)
    for t in range(0, ntiles):
        i = t * BLOCK_SCAN + lane
        slot = tl.where(direction == 0, i, ntiles * BLOCK_SCAN - 1 - i)
        row = block_row + slot
        pos = pos0 + tl.load(Perm + row)
        valid = pos < seqlen
        neg_f = -tl.load(F + (off_seq + pos) * 2 + f_col, mask=valid, other=0.0)
        neg_lse = -tl.load(Lse + off_seq + pos, mask=valid, other=0.0)
        hi, lo = _two_sum(neg_f, neg_lse)  # lw = hi + lo exactly, minus the lse residual
        lo = lo - tl.load(LseErr + off_seq + pos, mask=valid, other=0.0)
        # A query with f = +inf on this branch (masked: its branch-k logits are -inf) has weight 0. Its TwoSum would be
        # bb = (-inf) - (-inf) = NaN, and a NaN weight poisons every state of the block that includes it. (hi <= NEG
        # had weight exp(hi - ref_m) = 0 anyway, as ref_m >= NEG; a NaN hi, from a NaN f, still propagates.)
        ok = valid & ~(hi <= _NEG)
        lo = tl.where(ok, lo, 0.0)
        gm = tl.where(ok, hi, _NEG)
        ref_m, ref_m_prev = _running_references(gm, carry_m, BLOCK_SCAN)
        rescale = tl.exp(ref_m_prev - ref_m)
        weight = tl.where(ok, tl.exp((gm - ref_m) + lo), 0.0)
        do_m = tl.load(
            DO + (off_seq + pos)[:, None] * HEADDIM + offs_d[None, :],
            mask=valid[:, None] & mask_d[None, :],
            other=0.0,
        )
        weight_delta = weight * tl.load(Delta + off_seq + pos, mask=valid, other=0.0)
        S, z, carry_m, carry_S, carry_S_err, carry_z, carry_z_err = _scan_tile(
            rescale,
            weight,
            do_m,
            weight_delta,
            ref_m,
            carry_m,
            carry_S,
            carry_S_err,
            carry_z,
            carry_z_err,
            is_last,
            BLOCK_SCAN,
            BLOCK_HEADDIM,
        )
        tl.store(S_out + row[:, None] * HEADDIM + offs_d[None, :], S, mask=mask_d[None, :])
        if start_d == 0:
            tl.store(MZ_out + row * 2, ref_m)
            tl.store(MZ_out + row * 2 + 1, z)


@triton.jit
def _bwd_search_kernel(
    G,
    Rank,
    Split,
    seqlen,
    batch_heads,
    nchunks,
    nlevels,
    CHUNK: tl.constexpr,
    LOG2_CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_LEVELS: tl.constexpr,
):
    """Split[bh, l, y] = #queries with a(x) < r(y) in the level-l query block read by key y (0 where it reads none).

    Key y of chunk c reads level l iff bit l of c is 0 and the odd block after its even level-l block exists. The
    comparison is strict because the forward puts y on branch 1 of x iff r(y) <= a(x): slots >= split are y's
    branch-1 queries, slots < split its branch-2 queries.
    """
    start_n, off_bh = _split_program_id(seqlen, BLOCK_N)
    off_seq = off_bh.to(tl.int64) * seqlen
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seqlen
    chunk_idx = offs_n // CHUNK
    krank = tl.load(G + (off_seq + offs_n) * 2, mask=mask_n, other=0.0) - tl.load(
        G + (off_seq + offs_n) * 2 + 1, mask=mask_n, other=0.0
    )
    levels = tl.arange(0, BLOCK_LEVELS)
    nblocks = _num_blocks(nchunks, levels)  # garbage (unused) for levels >= nlevels
    block_idx = chunk_idx[:, None] >> (levels[None, :] + 1)
    reads = (
        ((((chunk_idx[:, None] >> levels[None, :]) & 1) == 0) & (levels < nlevels)[None, :] & mask_n[:, None])
        & (block_idx < nblocks[None, :])
    )
    block_len = CHUNK << levels
    block_row = _level_row_offset(levels, batch_heads, nchunks, nlevels, CHUNK)[None, :] + (
        off_bh.to(tl.int64) * nblocks[None, :] + block_idx
    ) * block_len[None, :]
    split = tl.zeros((BLOCK_N, BLOCK_LEVELS), tl.int32)
    step = 1 << (LOG2_CHUNK + nlevels - 1)
    for _ in range(0, LOG2_CHUNK + nlevels):
        probe_pos = split + step
        in_range = reads & (probe_pos <= block_len[None, :])
        probe = tl.load(Rank + block_row + (probe_pos - 1), mask=in_range, other=0.0)
        split = tl.where(in_range & (probe < krank[:, None]), probe_pos, split)
        step = step // 2
    tl.store(
        Split + (off_bh.to(tl.int64) * nlevels + levels[None, :]) * seqlen + offs_n[:, None],
        split,
        mask=mask_n[:, None] & (levels < nlevels)[None, :],
    )


@triton.jit
def _bwd_intra_kernel(
    F,
    G,
    Lse,
    LseErr,
    Delta,
    DO,
    V,
    DV,
    DG,
    seqlen,
    CHUNK: tl.constexpr,
    HEADDIM: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """dV (one headdim block) and dg1 / dg2 of BLOCK_N keys of one chunk from the queries of the same chunk (dense
    tile); writes DV (adds to it with ACCUMULATE) and writes DG.

    FlashAttention style in the transposed (keys x queries) orientation: P^T = exp(logit - lse) (branch by the rank
    rule), dP^T = V dO^T, dS^T = P^T (dP^T - D); dV = P^T dO, dg_k = row sums of dS^T over the branch-k queries. When
    one block holds all of headdim (BLOCK_HEADDIM >= HEADDIM) V^T is loaded once; otherwise every headdim-block program
    recomputes dP^T over all of headdim in BLOCK_HEADDIM-wide steps and block 0 writes DG. For a mixture component,
    LseErr is the preprocess kernel's lerr - log w: P^T is then w P^T, which scales dS^T and P^T dO alike (the backward
    for the upstream gradient w dO).
    """
    start_n, off_bh = _split_program_id(seqlen, BLOCK_N)
    start_d = tl.program_id(1)
    off_seq = off_bh.to(tl.int64) * seqlen
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seqlen
    mask_key = mask_n & (offs_n >= 1)  # key 0 is the sink: no order logit
    chunk_idx = (start_n * BLOCK_N) // CHUNK
    offs_d = start_d * BLOCK_HEADDIM + tl.arange(0, BLOCK_HEADDIM)
    mask_d = offs_d < HEADDIM
    g1_n = tl.load(G + (off_seq + offs_n) * 2, mask=mask_n, other=0.0)
    g2_n = tl.load(G + (off_seq + offs_n) * 2 + 1, mask=mask_n, other=0.0)
    krank = g1_n - g2_n
    v_n = tl.load(
        V + (off_seq + offs_n)[:, None] * HEADDIM + offs_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=0.0
    ).to(tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_HEADDIM), tl.float32)
    dg1 = tl.zeros((BLOCK_N,), tl.float32)
    dg2 = tl.zeros((BLOCK_N,), tl.float32)
    # queries before start_n BLOCK_N cannot see these keys, those after the chunk use the tree
    end_m = tl.minimum((chunk_idx + 1) * CHUNK, seqlen)
    for start_m in range((start_n * BLOCK_N // BLOCK_M) * BLOCK_M, end_m, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < end_m
        f1_m = tl.load(F + (off_seq + offs_m) * 2, mask=mask_m, other=0.0)
        f2_m = tl.load(F + (off_seq + offs_m) * 2 + 1, mask=mask_m, other=0.0)
        lse_m = tl.load(Lse + off_seq + offs_m, mask=mask_m, other=0.0)
        lse_err_m = tl.load(LseErr + off_seq + offs_m, mask=mask_m, other=0.0)
        delta_m = tl.load(Delta + off_seq + offs_m, mask=mask_m, other=0.0)
        branch1 = krank[:, None] <= (f1_m - f2_m)[None, :]  # (BLOCK_N, BLOCK_M)
        logit = tl.where(branch1, g1_n[:, None] - f1_m[None, :], g2_n[:, None] - f2_m[None, :])
        visible = (offs_n[:, None] <= offs_m[None, :]) & mask_key[:, None] & mask_m[None, :]
        p_t = tl.where(visible, tl.exp((logit - lse_m[None, :]) - lse_err_m[None, :]), 0.0)
        do_m = tl.load(
            DO + (off_seq + offs_m)[:, None] * HEADDIM + offs_d[None, :],
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        )
        if BLOCK_HEADDIM >= HEADDIM:
            dp_t = tl.dot(v_n, tl.trans(do_m), input_precision="ieee")
        else:
            dp_t = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
            for d_lo in range(0, HEADDIM, BLOCK_HEADDIM):
                offs_dd = d_lo + tl.arange(0, BLOCK_HEADDIM)
                v_n_all = tl.load(
                    V + (off_seq + offs_n)[:, None] * HEADDIM + offs_dd[None, :],
                    mask=mask_n[:, None] & (offs_dd < HEADDIM)[None, :],
                    other=0.0,
                ).to(tl.float32)
                do_m_all = tl.load(
                    DO + (off_seq + offs_m)[:, None] * HEADDIM + offs_dd[None, :],
                    mask=mask_m[:, None] & (offs_dd < HEADDIM)[None, :],
                    other=0.0,
                )
                dp_t += tl.dot(v_n_all, tl.trans(do_m_all), input_precision="ieee")
        ds_t = p_t * (dp_t - delta_m[None, :])
        dg1 += tl.sum(tl.where(branch1, ds_t, 0.0), axis=1)
        dg2 += tl.sum(tl.where(branch1, 0.0, ds_t), axis=1)
        acc_dv += tl.dot(p_t, do_m, input_precision="ieee")
    if ACCUMULATE:  # after the loop: loading the old dV into acc_dv before it made ptxas spill (intra 1.27 -> 4.76 ms)
        acc_dv += tl.load(
            DV + (off_seq + offs_n)[:, None] * HEADDIM + offs_d[None, :],
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
    tl.store(
        DV + (off_seq + offs_n)[:, None] * HEADDIM + offs_d[None, :], acc_dv, mask=mask_n[:, None] & mask_d[None, :]
    )
    if start_d == 0:
        tl.store(DG + (off_seq + offs_n) * 2, dg1, mask=mask_n)
        tl.store(DG + (off_seq + offs_n) * 2 + 1, dg2, mask=mask_n)


@triton.jit
def _bwd_tree_kernel(
    G,
    V,
    Split,
    S1,
    MZ1,
    S2,
    MZ2,
    DV,
    DG,
    seqlen,
    batch_heads,
    nchunks,
    nlevels,
    CHUNK: tl.constexpr,
    HEADDIM: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Adds the later chunks' share to DV (one headdim block) of BLOCK_N keys of one chunk and writes their dg1 / dg2
    parts.

    For every level l with bit l of the key's chunk clear, the odd sibling block's branch-1 suffix state at slot split
    and branch-2 prefix state at slot split - 1. Every term is a probability times an upstream quantity, so the sum is
    linear: a state (m, S, zD) of branch k contributes exp(g_k(y) + m) [S, <v(y), S> - zD], whose exponent is
    log p(x, y) of the state's largest query up to the whole-nat rounding of m (so <= ~1) and exact for the dominant
    terms (m is an integer). dg parts are per headdim block (the zD terms go to block 0) and summed afterwards.
    """
    start_n, off_bh = _split_program_id(seqlen, BLOCK_N)
    start_d = tl.program_id(1)
    off_seq = off_bh.to(tl.int64) * seqlen
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seqlen
    mask_key = mask_n & (offs_n >= 1)
    chunk_idx = (start_n * BLOCK_N) // CHUNK
    offs_d = start_d * BLOCK_HEADDIM + tl.arange(0, BLOCK_HEADDIM)
    mask_d = offs_d < HEADDIM
    g1_n = tl.load(G + (off_seq + offs_n) * 2, mask=mask_n, other=0.0)
    g2_n = tl.load(G + (off_seq + offs_n) * 2 + 1, mask=mask_n, other=0.0)
    dv_ptrs = DV + (off_seq + offs_n)[:, None] * HEADDIM + offs_d[None, :]
    mask_nd = mask_n[:, None] & mask_d[None, :]
    v_n = tl.load(V + (off_seq + offs_n)[:, None] * HEADDIM + offs_d[None, :], mask=mask_nd, other=0.0).to(tl.float32)
    acc_dv = tl.load(dv_ptrs, mask=mask_nd, other=0.0)  # the intra share, written by _bwd_intra_kernel
    dg1 = tl.zeros((BLOCK_N,), tl.float32)
    dg2 = tl.zeros((BLOCK_N,), tl.float32)
    is_first_block = (start_d == 0).to(tl.float32)
    level_row = off_bh.to(tl.int64) * 0  # first buffer row of the level (an int64 zero tensor, carried by the loop)
    for level in range(0, nlevels):
        block_len = CHUNK << level
        nblocks = _num_blocks(nchunks, level)
        block_idx = chunk_idx >> (level + 1)
        if (((chunk_idx >> level) & 1) == 0) & (block_idx < nblocks):
            nreal = tl.minimum(seqlen - (2 * block_idx + 1) * block_len, block_len)
            block_row = level_row + (off_bh.to(tl.int64) * nblocks + block_idx) * block_len
            split = tl.load(Split + (off_bh.to(tl.int64) * nlevels + level) * seqlen + offs_n, mask=mask_n, other=0)
            has1 = (split < nreal) & mask_key  # some branch-1 query: suffix state at slot split
            has2 = (split > 0) & mask_key  # some branch-2 query: prefix state at slot split - 1
            row1 = block_row + tl.minimum(split, nreal - 1)
            row2 = block_row + tl.maximum(split - 1, 0)
            state_m1 = tl.load(MZ1 + row1 * 2, mask=has1, other=0.0)
            state_z1 = tl.load(MZ1 + row1 * 2 + 1, mask=has1, other=0.0)
            state_m2 = tl.load(MZ2 + row2 * 2, mask=has2, other=0.0)
            state_z2 = tl.load(MZ2 + row2 * 2 + 1, mask=has2, other=0.0)
            state_S1 = tl.load(
                S1 + row1[:, None] * HEADDIM + offs_d[None, :], mask=has1[:, None] & mask_d[None, :], other=0.0
            )
            state_S2 = tl.load(
                S2 + row2[:, None] * HEADDIM + offs_d[None, :], mask=has2[:, None] & mask_d[None, :], other=0.0
            )
            scale1 = tl.where(has1, tl.exp(g1_n + state_m1), 0.0)
            scale2 = tl.where(has2, tl.exp(g2_n + state_m2), 0.0)
            acc_dv += scale1[:, None] * state_S1 + scale2[:, None] * state_S2
            dg1 += scale1 * (tl.sum(v_n * state_S1, axis=1) - is_first_block * state_z1)
            dg2 += scale2 * (tl.sum(v_n * state_S2, axis=1) - is_first_block * state_z2)
        level_row += (nblocks * block_len).to(tl.int64) * batch_heads
    tl.store(dv_ptrs, acc_dv, mask=mask_nd)
    dg_ptrs = DG + (start_d.to(tl.int64) * batch_heads * seqlen + off_seq + offs_n) * 2
    tl.store(dg_ptrs, dg1, mask=mask_n)
    tl.store(dg_ptrs + 1, dg2, mask=mask_n)


@triton.jit
def _mixture_sum_kernel(
    O,
    W,
    Out,
    nrows,
    ncomponents,
    first_component,
    stride_oc,
    HEADDIM: tl.constexpr,
    NUM_COMPONENTS: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    """Gate-weighted sum of mixture components: Out[r] = (Out[r] if ACCUMULATE else 0) + sum_{c < NUM_COMPONENTS}
    W[r, first_component + c] O[c, r] for BLOCK_M rows r of the (batch nheads seqlen) rows and one column block, with
    O (NUM_COMPONENTS, batch nheads seqlen, headdim) (component stride ``stride_oc``) and W (batch nheads seqlen,
    ncomponents), in the order c = 0 .. NUM_COMPONENTS - 1 (so NUM_COMPONENTS = ncomponents at once and ncomponents
    calls with NUM_COMPONENTS = 1 add the same terms in the same order)."""
    offs_m = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_d = tl.program_id(1) * BLOCK_HEADDIM + tl.arange(0, BLOCK_HEADDIM)
    mask_m = offs_m < nrows
    mask_md = mask_m[:, None] & (offs_d < HEADDIM)[None, :]
    ptrs = offs_m[:, None] * HEADDIM + offs_d[None, :]
    if ACCUMULATE:
        acc = tl.load(Out + ptrs, mask=mask_md, other=0.0)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_HEADDIM), tl.float32)
    o_component = O
    for c in tl.static_range(NUM_COMPONENTS):
        w_m = tl.load(W + offs_m * ncomponents + first_component + c, mask=mask_m, other=0.0)
        acc += w_m[:, None] * tl.load(o_component + ptrs, mask=mask_md, other=0.0)
        o_component += stride_oc  # pointer arithmetic is 64-bit: no int32 overflow of c * stride_oc
    tl.store(Out + ptrs, acc, mask=mask_md)


# ----------------------------------------------------------------------------------------------------------------------
# Python API: shared helpers
# ----------------------------------------------------------------------------------------------------------------------


def _num_level_blocks(nchunks: int) -> list[int]:
    """Blocks per level: nb_l = #even level-l blocks with a query chunk after them, l = 0 .. ceil(log2 nchunks) - 1."""
    out, level = [], 0
    while (1 << level) < nchunks:
        out.append((nchunks - (1 << level) - 1) // (2 << level) + 1)
        level += 1
    return out


def _level_layout(nblocks: Sequence[int], chunk: int, batch_heads: int) -> tuple[list[int], list[int], int]:
    """(rows per level, first row of each level, total rows) of the flat level buffers; the total is at least 1."""
    sizes = [batch_heads * nb * (chunk << level) for level, nb in enumerate(nblocks)]
    offsets = [sum(sizes[:level]) for level in range(len(nblocks))]
    return sizes, offsets, max(sum(sizes), 1)


def _empty_level_states(
    total: int, headdim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uninitialised (rank, S1, mz1, S2, mz2) buffers of ``total`` rows."""
    rank = torch.empty(total, device=device, dtype=torch.float32)
    S1 = torch.empty(total, headdim, device=device, dtype=torch.float32)
    S2 = torch.empty(total, headdim, device=device, dtype=torch.float32)
    mz1 = torch.empty(total, 2, device=device, dtype=torch.float32)
    mz2 = torch.empty(total, 2, device=device, dtype=torch.float32)
    return rank, S1, mz1, S2, mz2


def _check_inputs(f: torch.Tensor, g: torch.Tensor, b: torch.Tensor, v: torch.Tensor, chunk: int) -> None:
    if f.dim() != 4 or f.shape[-1] != 2 or g.shape != f.shape:
        raise ValueError(
            f"expected f, g of shape (batch, nheads, seqlen, 2), got {tuple(f.shape)} and {tuple(g.shape)}"
        )
    if b.shape != f.shape[:-1] or v.dim() != 4 or v.shape[:-1] != f.shape[:-1]:
        raise ValueError(
            f"expected b (batch, nheads, seqlen) and v (batch, nheads, seqlen, headdim), got {tuple(b.shape)} and "
            f"{tuple(v.shape)}"
        )
    if chunk < 16 or chunk & (chunk - 1):
        raise ValueError(f"chunk must be a power of two >= 16, got {chunk}")
    if f.shape[2] == 0:
        raise ValueError("empty sequence")
    if not f.is_cuda:
        if os.environ.get("TRITON_INTERPRET") != "1":
            raise ValueError("order_attn_func needs CUDA tensors (or TRITON_INTERPRET=1)")
        logger.debug("running the order attention kernels under the Triton interpreter on %s", f.device)


def _as_kernel_values(v: torch.Tensor) -> torch.Tensor:
    """``v`` as the kernels read it: fp32, bf16 or fp16 (any other dtype converted to fp32), contiguous."""
    if v.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        v = v.float()
    return v.contiguous()


def _on_device(t: torch.Tensor):
    """Makes ``t``'s CUDA device the current one (Triton launches on the current device); no-op off CUDA."""
    return torch.cuda.device(t.device) if t.is_cuda else contextlib.nullcontext()


# ----------------------------------------------------------------------------------------------------------------------
# Python API: forward
# ----------------------------------------------------------------------------------------------------------------------


def _fwd_build_levels(
    g: torch.Tensor, v: torch.Tensor, chunk: int, timer: Timer = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Sort every level block by rank and scan it into prefix / suffix states.

    Level l occupies rows lvl[l] .. lvl[l] + batch_heads nb_l Lb_l of every buffer, lvl[l] = batch_heads
    sum_{l' < l} nb_l' Lb_l' (nb_l from ``_num_level_blocks``; the kernels compute lvl[l] themselves), ordered
    (bh, block, sorted slot).

    Arguments:
        g: (batch, nheads, seqlen, 2) fp32 contiguous key realizers.
        v: (batch, nheads, seqlen, headdim) values, from ``_as_kernel_values``.
        chunk: keys per chunk.
        timer: called around the "sort" and "scan" phases.
    Return:
        rank: (rows,) fp32 sorted key ranks.
        S1, mz1: (rows, headdim) value sums and (rows, 2) [reference m, normaliser z] of the prefix states
            (m = ceil(running max), see ``_fwd_scan_kernel``).
        S2, mz2: the same for the suffix states.
        nlevels: the number of levels (0 for a single chunk).
    """
    batch, nheads, seqlen, _ = g.shape
    batch_heads, headdim = batch * nheads, v.shape[-1]
    nchunks = triton.cdiv(seqlen, chunk)
    nblocks = _num_level_blocks(nchunks)
    sizes, offsets, total = _level_layout(nblocks, chunk, batch_heads)
    device = g.device
    rank, S1, mz1, S2, mz2 = _empty_level_states(total, headdim, device)
    if not nblocks:
        return rank, S1, mz1, S2, mz2, 0

    npos = (1 << len(nblocks)) * chunk  # positions covered by the top level's block pair
    krank = (g[..., 0] - g[..., 1]).reshape(batch_heads, seqlen)
    if npos > seqlen:
        krank = torch.nn.functional.pad(krank, (0, npos - seqlen))
    perm = torch.empty(total, device=device, dtype=torch.int64)
    if timer:
        timer("sort", True)
    for level, nb in enumerate(nblocks):
        block_len = chunk << level
        src = krank[:, :npos].view(batch_heads, npos // (2 * block_len), 2, block_len)[:, :nb, 0, :]  # even blocks
        rows = slice(offsets[level], offsets[level] + sizes[level])
        torch.sort(
            src, dim=-1, out=(rank[rows].view(batch_heads, nb, block_len), perm[rows].view(batch_heads, nb, block_len))
        )
    if timer:
        timer("sort", False)
        timer("scan", True)
    block_headdim = min(SCAN_BLOCK_HEADDIM, max(16, triton.next_power_of_2(headdim)))
    num_headdim_blocks = triton.cdiv(headdim, block_headdim)
    num_warps = SCAN_NUM_WARPS if v.element_size() == 4 else SCAN_NUM_WARPS_LOWP
    _fwd_scan_kernel[(sum(nblocks) * batch_heads * 2 * num_headdim_blocks,)](
        perm,
        g,
        v,
        S1,
        mz1,
        S2,
        mz2,
        seqlen,
        batch_heads,
        nchunks,
        len(nblocks),
        CHUNK=chunk,
        HEADDIM=headdim,
        BLOCK_HEADDIM=block_headdim,
        NUM_HEADDIM_BLOCKS=num_headdim_blocks,
        BLOCK_SCAN=min(SCAN_BLOCK, chunk),
        num_warps=num_warps,
    )
    if timer:
        timer("scan", False)
    return rank, S1, mz1, S2, mz2, len(nblocks)


def _order_attn_forward(
    f: torch.Tensor, g: torch.Tensor, b: torch.Tensor, v: torch.Tensor, chunk: int = 64, timer: Timer = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass: checks the inputs, then runs the kernels on their device.

    No host synchronisation and no host-to-device copy (the kernels compute the level offsets themselves), and a
    captured graph references only its own buffers, so the forward is CUDA-graph capturable once its kernels are
    compiled (also on the first call with a new shape).

    Arguments:
        f: (batch, nheads, seqlen, 2) query realizers (f1, f2).
        g: (batch, nheads, seqlen, 2) key realizers (g1, g2).
        b: (batch, nheads, seqlen) sink logits.
        v: (batch, nheads, seqlen, headdim) values.
        chunk: keys per chunk (a power of two >= 16).
        timer: called around the phases "sort", "scan", "search" and "query" (benchmarks).
    Return:
        out: (batch, nheads, seqlen, headdim) fp32 attention output.
        softmax_lse: (batch, nheads, seqlen) fp32 log-sum-exp of the logits of every query.
        branch1_mass: (batch, nheads, seqlen) fp32 A1, the probability mass on branch 1.
        branch1_out: (batch, nheads, seqlen, headdim) fp32 U1, the branch-1 part of the output.
        sink_prob: (batch, nheads, seqlen) fp32 p0, the sink probability.
        lse_residual: (batch, nheads, seqlen) fp32 TwoSum residual of softmax_lse (see the module docstring).
    """
    _check_inputs(f, g, b, v, chunk)
    with _on_device(f):
        return _order_attn_forward_impl(f, g, b, v, chunk, timer)


def _order_attn_forward_impl(
    f: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    chunk: int,
    timer: Timer,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_order_attn_forward`` on the current device, writing the output into ``out`` (contiguous fp32) if given."""
    batch, nheads, seqlen, _ = f.shape
    headdim = v.shape[-1]
    f = f.float().contiguous()
    g = g.float().contiguous()
    b = b.float().contiguous()
    v = _as_kernel_values(v)
    rank, S1, mz1, S2, mz2, nlevels = _fwd_build_levels(g, v, chunk, timer)
    nchunks = triton.cdiv(seqlen, chunk)
    batch_heads = batch * nheads
    split = torch.empty(batch_heads * max(nlevels, 1) * seqlen, device=f.device, dtype=torch.int32)
    if timer:
        timer("search", True)
    if nlevels:
        _fwd_search_kernel[(triton.cdiv(seqlen, SEARCH_BLOCK_M) * batch_heads,)](
            f,
            rank,
            split,
            seqlen,
            batch_heads,
            nchunks,
            nlevels,
            CHUNK=chunk,
            LOG2_CHUNK=int(math.log2(chunk)),
            BLOCK_M=SEARCH_BLOCK_M,
            BLOCK_LEVELS=max(2, triton.next_power_of_2(nlevels)),
            num_warps=4,
        )
    if timer:
        timer("search", False)
        timer("query", True)
    if out is None:
        out = torch.empty(batch, nheads, seqlen, headdim, device=f.device, dtype=torch.float32)
    branch1_out = torch.empty_like(out)
    softmax_lse = torch.empty(batch, nheads, seqlen, device=f.device, dtype=torch.float32)
    lse_residual = torch.empty_like(softmax_lse)
    branch1_mass = torch.empty_like(softmax_lse)
    sink_prob = torch.empty_like(softmax_lse)
    block_headdim = min(FWD_BLOCK_HEADDIM, max(16, triton.next_power_of_2(headdim)))  # tl.dot needs every dim >= 16
    _fwd_kernel[(triton.cdiv(seqlen, FWD_BLOCK_M) * batch_heads, triton.cdiv(headdim, block_headdim))](
        f,
        g,
        b,
        v,
        split,
        S1,
        mz1,
        S2,
        mz2,
        out,
        softmax_lse,
        lse_residual,
        branch1_mass,
        branch1_out,
        sink_prob,
        seqlen,
        batch_heads,
        nchunks,
        nlevels,
        CHUNK=chunk,
        HEADDIM=headdim,
        BLOCK_HEADDIM=block_headdim,
        BLOCK_M=FWD_BLOCK_M,
        BLOCK_N=FWD_BLOCK_M,
        num_warps=4,
    )
    if timer:
        timer("query", False)
    return out, softmax_lse, branch1_mass, branch1_out, sink_prob, lse_residual


# ----------------------------------------------------------------------------------------------------------------------
# Python API: backward
# ----------------------------------------------------------------------------------------------------------------------


def _bwd_build_levels(
    f: torch.Tensor,
    softmax_lse: torch.Tensor,
    lse_residual: torch.Tensor,
    dout: torch.Tensor,
    delta: torch.Tensor,
    chunk: int,
    timer: Timer = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Sort every odd level block of queries by rank a(x) and scan it into the backward's suffix / prefix states.

    Same row layout as ``_fwd_build_levels`` (level l, bh, block k, sorted slot; nb_l blocks of Lb_l = chunk 2^l
    slots), but block k of level l holds the queries of positions (2k + 1) Lb_l .. (2k + 2) Lb_l - 1; padding
    positions (>= seqlen) get rank +inf and sort last.

    Arguments:
        f: (batch, nheads, seqlen, 2) fp32 contiguous query realizers.
        softmax_lse, lse_residual: (batch, nheads, seqlen) from the forward (the residual possibly folded with
            -log w for a mixture component, see ``_bwd_preprocess_kernel``).
        dout: (batch, nheads, seqlen, headdim) fp32 contiguous upstream gradient.
        delta: (batch, nheads, seqlen) fp32 D = <dO, out>.
        chunk: keys per chunk.
        timer: called around the "sort" and "scan" phases.
    Return:
        rank, S1, mz1, S2, mz2, nlevels: as ``_fwd_build_levels``, with S1 / mz1 the branch-1 SUFFIX states and
            S2 / mz2 the branch-2 PREFIX states of the payload [dO, D], see ``_bwd_scan_kernel``.
    """
    batch, nheads, seqlen, _ = f.shape
    batch_heads, headdim = batch * nheads, dout.shape[-1]
    nchunks = triton.cdiv(seqlen, chunk)
    nblocks = _num_level_blocks(nchunks)
    sizes, offsets, total = _level_layout(nblocks, chunk, batch_heads)
    device = f.device
    rank, S1, mz1, S2, mz2 = _empty_level_states(total, headdim, device)
    if not nblocks:
        return rank, S1, mz1, S2, mz2, 0

    npos = (1 << len(nblocks)) * chunk
    qrank = (f[..., 0] - f[..., 1]).reshape(batch_heads, seqlen)  # bitwise the a(x) of the forward kernels
    if npos > seqlen:
        qrank = torch.nn.functional.pad(qrank, (0, npos - seqlen), value=float("inf"))
    perm = torch.empty(total, device=device, dtype=torch.int64)
    if timer:
        timer("sort", True)
    for level, nb in enumerate(nblocks):
        block_len = chunk << level
        src = qrank[:, :npos].view(batch_heads, npos // (2 * block_len), 2, block_len)[:, :nb, 1, :]  # odd blocks
        rows = slice(offsets[level], offsets[level] + sizes[level])
        torch.sort(
            src, dim=-1, out=(rank[rows].view(batch_heads, nb, block_len), perm[rows].view(batch_heads, nb, block_len))
        )
    if timer:
        timer("sort", False)
        timer("scan", True)
    block_headdim = min(BWD_SCAN_BLOCK_HEADDIM, max(16, triton.next_power_of_2(headdim)))
    num_headdim_blocks = triton.cdiv(headdim, block_headdim)
    _bwd_scan_kernel[(sum(nblocks) * batch_heads * 2 * num_headdim_blocks,)](
        perm,
        f,
        softmax_lse,
        lse_residual,
        dout,
        delta,
        S1,
        mz1,
        S2,
        mz2,
        seqlen,
        batch_heads,
        nchunks,
        len(nblocks),
        CHUNK=chunk,
        HEADDIM=headdim,
        BLOCK_HEADDIM=block_headdim,
        NUM_HEADDIM_BLOCKS=num_headdim_blocks,
        BLOCK_SCAN=min(SCAN_BLOCK, chunk),
        num_warps=BWD_SCAN_NUM_WARPS,
    )
    if timer:
        timer("scan", False)
    return rank, S1, mz1, S2, mz2, len(nblocks)


def _order_attn_backward(
    dout: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    branch1_mass: torch.Tensor,
    branch1_out: torch.Tensor,
    sink_prob: torch.Tensor,
    lse_residual: torch.Tensor,
    chunk: int = 64,
    timer: Timer = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass from the forward's saved tensors (``_order_attn_forward``, in its return order).

    ``b`` is not read (db comes from the saved sink probability); it is taken to mirror the forward's arguments. No
    host synchronisation and no host-to-device copy.

    Arguments:
        dout: (batch, nheads, seqlen, headdim) upstream gradient.
        f, g, b, v: the forward's inputs.
        out, softmax_lse, branch1_mass, branch1_out, sink_prob, lse_residual: the forward's outputs.
        chunk: keys per chunk, as in the forward.
        timer: called around the phases "prep", "sort", "scan", "search", "intra" and "tree" (benchmarks).
    Return:
        df, dg: (batch, nheads, seqlen, 2) fp32 gradients of the realizers.
        db: (batch, nheads, seqlen) fp32 gradient of the sink logits.
        dv: (batch, nheads, seqlen, headdim) fp32 gradient of the values.
    """
    with _on_device(f):
        batch, nheads, seqlen, _ = f.shape
        headdim = v.shape[-1]
        dv = torch.empty(batch, nheads, seqlen, headdim, device=f.device, dtype=torch.float32)
        dv_sink = torch.empty(
            batch * nheads, triton.cdiv(seqlen, BWD_PREPROCESS_BLOCK_M), headdim, device=f.device, dtype=torch.float32
        )
        df, dg, db = _order_attn_backward_impl(
            dout.float().contiguous(),
            f.float().contiguous(),
            g.float().contiguous(),
            _as_kernel_values(v),
            out,
            softmax_lse,
            branch1_mass,
            branch1_out,
            sink_prob,
            lse_residual,
            chunk,
            dv,
            dv_sink,
            timer,
        )
        dv[:, :, 0] = dv_sink.sum(1).view(batch, nheads, headdim)  # key 0 is only ever the sink
        return df, dg, db, dv


def _order_attn_backward_impl(
    dout: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    branch1_mass: torch.Tensor,
    branch1_out: torch.Tensor,
    sink_prob: torch.Tensor,
    lse_residual: torch.Tensor,
    chunk: int,
    dv: torch.Tensor,
    dv_sink: torch.Tensor,
    timer: Timer,
    accumulate: bool = False,
    gate_weights: Optional[torch.Tensor] = None,
    component: int = 0,
    dgate_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The backward on the current device (dout, f, g fp32 contiguous, v from ``_as_kernel_values``).

    Writes the order keys' value gradients into ``dv`` (rows y >= 1; adds to ``dv`` with ``accumulate``) and leaves
    row 0 unchanged (zero without ``accumulate``); writes the parts of the sink's value gradient sum_x p0(x) dO(x), one
    per BWD_PREPROCESS_BLOCK_M queries, into ``dv_sink`` (batch nheads, ceil(seqlen / BWD_PREPROCESS_BLOCK_M), headdim)
    for the caller to sum. With ``gate_weights`` (batch, nheads, seqlen, ncomponents) this is the backward of mixture
    component ``component`` for the upstream gradient gate_weights[..., component] dO, without forming it: the
    preprocess kernel multiplies df, db and dV(0) by the weight and hands the scan and intra kernels lerr - log w in
    place of lerr (see ``_bwd_preprocess_kernel``), and it writes dL/dw[..., component] = <dO, out> into
    ``dgate_weights[..., component]``.

    Return:
        df, dg: (batch, nheads, seqlen, 2) fp32.
        db: (batch, nheads, seqlen) fp32.
    """
    batch, nheads, seqlen, _ = f.shape
    batch_heads, headdim = batch * nheads, v.shape[-1]
    device = f.device
    nchunks = triton.cdiv(seqlen, chunk)
    headdim_pow2 = max(16, triton.next_power_of_2(headdim))  # tl.dot needs every dim >= 16
    is_mixture = gate_weights is not None

    # per query: D, df1, df2, db; per program: a part of the sink's value gradient sum_x p0(x) dO(x)
    if timer:
        timer("prep", True)
    delta = torch.empty(batch, nheads, seqlen, device=device, dtype=torch.float32)
    df = torch.empty(batch, nheads, seqlen, 2, device=device, dtype=torch.float32)
    db = torch.empty(batch, nheads, seqlen, device=device, dtype=torch.float32)
    # the residual the scan and intra kernels read
    lse_residual_k = torch.empty_like(lse_residual) if is_mixture else lse_residual
    _bwd_preprocess_kernel[(triton.cdiv(seqlen, BWD_PREPROCESS_BLOCK_M) * batch_heads,)](
        dout,
        out,
        branch1_out,
        v,
        branch1_mass,
        sink_prob,
        delta,
        df,
        db,
        dv_sink,
        gate_weights if is_mixture else delta,
        dgate_weights if is_mixture else delta,
        lse_residual,
        lse_residual_k,
        seqlen,
        gate_weights.shape[-1] if is_mixture else 1,
        component,
        HEADDIM=headdim,
        BLOCK_HEADDIM=min(64, headdim_pow2),
        BLOCK_M=BWD_PREPROCESS_BLOCK_M,
        IS_MIXTURE=is_mixture,
        num_warps=4,
    )
    if timer:
        timer("prep", False)

    rank, S1, mz1, S2, mz2, nlevels = _bwd_build_levels(f, softmax_lse, lse_residual_k, dout, delta, chunk, timer)
    split = torch.empty(batch_heads * max(nlevels, 1) * seqlen, device=device, dtype=torch.int32)
    if timer:
        timer("search", True)
    if nlevels:
        _bwd_search_kernel[(triton.cdiv(seqlen, SEARCH_BLOCK_M) * batch_heads,)](
            g,
            rank,
            split,
            seqlen,
            batch_heads,
            nchunks,
            nlevels,
            CHUNK=chunk,
            LOG2_CHUNK=int(math.log2(chunk)),
            BLOCK_N=SEARCH_BLOCK_M,
            BLOCK_LEVELS=max(2, triton.next_power_of_2(nlevels)),
            num_warps=4,
        )
    if timer:
        timer("search", False)
        timer("intra", True)
    tree_block_headdim = min(BWD_TREE_BLOCK_HEADDIM, headdim_pow2)
    num_tree_blocks = triton.cdiv(headdim, tree_block_headdim) if nlevels else 0
    dg_parts = torch.empty(1 + num_tree_blocks, batch, nheads, seqlen, 2, device=device, dtype=torch.float32)
    intra_block_n = min(BWD_INTRA_BLOCK_N, chunk)
    intra_block_headdim = headdim_pow2 if headdim_pow2 <= BWD_INTRA_FULL_HEADDIM else BWD_INTRA_BLOCK_HEADDIM
    _bwd_intra_kernel[(triton.cdiv(seqlen, intra_block_n) * batch_heads, triton.cdiv(headdim, intra_block_headdim))](
        f,
        g,
        softmax_lse,
        lse_residual_k,
        delta,
        dout,
        v,
        dv,
        dg_parts[0],
        seqlen,
        CHUNK=chunk,
        HEADDIM=headdim,
        BLOCK_HEADDIM=intra_block_headdim,
        BLOCK_N=intra_block_n,
        BLOCK_M=min(BWD_INTRA_BLOCK_M, chunk),
        ACCUMULATE=accumulate,
        num_warps=4,
        num_stages=1,
    )
    if timer:
        timer("intra", False)
        timer("tree", True)
    if nlevels:
        _bwd_tree_kernel[(triton.cdiv(seqlen, BWD_TREE_BLOCK_N) * batch_heads, num_tree_blocks)](
            g,
            v,
            split,
            S1,
            mz1,
            S2,
            mz2,
            dv,
            dg_parts[1:],
            seqlen,
            batch_heads,
            nchunks,
            nlevels,
            CHUNK=chunk,
            HEADDIM=headdim,
            BLOCK_HEADDIM=tree_block_headdim,
            BLOCK_N=BWD_TREE_BLOCK_N,
            num_warps=BWD_TREE_NUM_WARPS,
        )
    if timer:
        timer("tree", False)
    dg = dg_parts[0] if num_tree_blocks == 0 else dg_parts.sum(0)
    return df, dg, db


def _no_double_backward(backward: Callable) -> Callable:
    """Marks an autograd backward that is not differentiable itself (a Triton kernel chain): it raises when called
    under ``create_graph=True`` (inside a backward, grad mode is on exactly then).

    ``torch.autograd.function.once_differentiable`` is not enough: it attaches its error node only when the upstream
    gradient requires grad, so for a loss linear in the output (``out.sum()``, ``<c, out>``) the returned gradients
    were plain tensors and a gradient penalty, ``functional.hvp`` or ``functional.hessian`` silently got no
    second-order term (zeros); and ``autograd.grad`` with respect to the inputs prunes that node even when it is there.
    """

    @functools.wraps(backward)
    def wrapper(ctx, *grads):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "order_attn_func / mixture_attn_func have no double backward (the backward is a Triton kernel chain, "
                "not differentiable itself): differentiate without create_graph=True, or use the torch "
                "learning.order_attention.order_attention / mixture_attention for higher-order gradients"
            )
        return backward(ctx, *grads)  # grad mode is off: nothing is recorded

    return wrapper


# ----------------------------------------------------------------------------------------------------------------------
# Python API: autograd functions and entry points
# ----------------------------------------------------------------------------------------------------------------------


class OrderAttnFunc(torch.autograd.Function):
    """Autograd wrapper: the forward saves (out, softmax_lse, branch1_mass, branch1_out, sink_prob, lse_residual) next
    to the inputs, the backward reads them."""

    @staticmethod
    def forward(ctx, f, g, b, v, chunk):
        saved = _order_attn_forward(f, g, b, v, chunk)
        ctx.save_for_backward(f, g, b, v, *saved)
        ctx.chunk = chunk
        return saved[0]

    @staticmethod
    @_no_double_backward
    def backward(ctx, dout):
        f, g, b, v, *saved = ctx.saved_tensors
        df, dg, db, dv = _order_attn_backward(dout, f, g, b, v, *saved, ctx.chunk)
        return df.to(f.dtype), dg.to(g.dtype), db.to(b.dtype), dv.to(v.dtype), None


def order_attn_func(
    f: torch.Tensor, g: torch.Tensor, b: torch.Tensor, v: torch.Tensor, chunk: int = 64
) -> torch.Tensor:
    """Causal softmax(order scores) @ v, equal to ``order_attention(f, g, b, v, causal=True)``.

    Arguments:
        f: (batch, nheads, seqlen, 2) query realizer values (f1, f2).
        g: (batch, nheads, seqlen, 2) key realizer values (g1, g2); position 0 is the sink key and is scored by ``b``
            instead.
        b: (batch, nheads, seqlen) sink logit per query.
        v: (batch, nheads, seqlen, headdim) values, fp32 / bf16 / fp16 (accumulated in fp32).
        chunk: keys per chunk (a power of two >= 16): the dense intra-chunk tile and the leaves of the tree.
    Return:
        out: (batch, nheads, seqlen, headdim) fp32 attention output, differentiable in f, g, b and v (Triton backward,
            same chunked tree).
    """
    return OrderAttnFunc.apply(f, g, b, v, chunk)


order_attention_triton = order_attn_func  # backward-compatible name


def _check_mixture_inputs(gates: torch.Tensor, components: Sequence[Component], v: torch.Tensor, chunk: int) -> None:
    if not components:
        raise ValueError("a mixture needs at least one component")
    for comp in components:
        if len(comp) != 3:
            raise ValueError(f"expected components (f, g, b), got a tuple of {len(comp)}")
        _check_inputs(*comp, v, chunk)
    want = (*v.shape[:-1], len(components))
    if tuple(gates.shape) != want:
        raise ValueError(
            f"expected gates of shape (batch, nheads, seqlen, ncomponents) = {want}, got {tuple(gates.shape)}"
        )


def _mixture_attn_forward(
    gates: torch.Tensor, components: Sequence[Component], v: torch.Tensor, chunk: int, keep: bool
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[list[tuple[torch.Tensor, ...]]]]:
    """Mixture forward on the current device.

    Arguments:
        gates: (batch, nheads, seqlen, ncomponents) mixture logits.
        components: the (f, g, b) of every component.
        v: (batch, nheads, seqlen, headdim) shared values.
        chunk: keys per chunk.
        keep: keep every component's output and saved tensors for a backward. With ``keep`` every component writes
            its output into component_outs[m] of one (ncomponents, batch, nheads, seqlen, headdim) buffer, saved[m]
            holds its (softmax_lse, branch1_mass, branch1_out, sink_prob, lse_residual), and one
            ``_mixture_sum_kernel`` launch forms sum_m w_m component_outs[m]. Without (no gradient needed), each
            component's output is added into ``out`` and dropped, so only one component's tensors are alive at a time
            (component_outs and saved are None).
    Return:
        out: (batch, nheads, seqlen, headdim) fp32 mixture output.
        gate_weights: (batch, nheads, seqlen, ncomponents) fp32 w = softmax(gates).
        component_outs, saved: see ``keep``.
    """
    batch, nheads, seqlen, headdim = v.shape
    ncomponents = len(components)
    nrows = batch * nheads * seqlen
    gate_weights = torch.softmax(gates.float(), dim=-1).contiguous()
    v = _as_kernel_values(v)
    out = torch.empty(batch, nheads, seqlen, headdim, device=v.device, dtype=torch.float32)
    grid = (triton.cdiv(nrows, MIX_BLOCK_M), triton.cdiv(headdim, MIX_BLOCK_HEADDIM))
    if keep:
        component_outs = torch.empty(ncomponents, batch, nheads, seqlen, headdim, device=v.device, dtype=torch.float32)
        saved = [
            _order_attn_forward_impl(f, g, b, v, chunk, None, out=component_outs[m])[1:]
            for m, (f, g, b) in enumerate(components)
        ]
        _mixture_sum_kernel[grid](
            component_outs,
            gate_weights,
            out,
            nrows,
            ncomponents,
            0,
            nrows * headdim,
            HEADDIM=headdim,
            NUM_COMPONENTS=ncomponents,
            ACCUMULATE=False,
            BLOCK_M=MIX_BLOCK_M,
            BLOCK_HEADDIM=MIX_BLOCK_HEADDIM,
            num_warps=4,
        )
        return out, gate_weights, component_outs, saved
    component_out = torch.empty(batch, nheads, seqlen, headdim, device=v.device, dtype=torch.float32)  # reused
    for m, (f, g, b) in enumerate(components):
        _order_attn_forward_impl(f, g, b, v, chunk, None, out=component_out)
        _mixture_sum_kernel[grid](
            component_out,
            gate_weights,
            out,
            nrows,
            ncomponents,
            m,
            nrows * headdim,
            HEADDIM=headdim,
            NUM_COMPONENTS=1,
            ACCUMULATE=m > 0,
            BLOCK_M=MIX_BLOCK_M,
            BLOCK_HEADDIM=MIX_BLOCK_HEADDIM,
            num_warps=4,
        )
    return out, gate_weights, None, None


class MixtureAttnFunc(torch.autograd.Function):
    """Autograd wrapper of the gated mixture (``mixture_attn_func``), fused over its components.

    The forward saves the component outputs (one (ncomponents, batch, nheads, seqlen, headdim) buffer), each
    component's (softmax_lse, branch1_mass, branch1_out, sink_prob, lse_residual) and w = softmax(gates). The backward
    passes the upstream gradient once: component m's backward runs on it with every per-query term weighted by w_m
    (the preprocess kernel multiplies df, db and dV(0) by w_m and hands the scan and intra kernels lerr - log w_m in
    place of lerr), which is its backward for the upstream gradient w_m dO without forming that tensor. All components
    add into one dv (the intra kernel with ACCUMULATE, the tree kernel always adds), and dgates follows from
    dL/dw_m = <dO, out_m> (written by the preprocess kernel) through the softmax Jacobian.
    """

    @staticmethod
    def forward(ctx, gates, v, chunk, *flat):
        components = [flat[i : i + 3] for i in range(0, len(flat), 3)]
        with _on_device(v):
            out, gate_weights, component_outs, saved = _mixture_attn_forward(gates, components, v, chunk, keep=True)
        ctx.save_for_backward(v, gate_weights, component_outs, *flat, *[t for s in saved for t in s])
        ctx.chunk, ctx.ncomponents, ctx.gates_dtype = chunk, len(components), gates.dtype
        return out

    @staticmethod
    @_no_double_backward
    def backward(ctx, dout):
        ncomponents, chunk = ctx.ncomponents, ctx.chunk
        v, gate_weights, component_outs, *rest = ctx.saved_tensors
        flat, saved = rest[: 3 * ncomponents], rest[3 * ncomponents :]
        batch, nheads, seqlen, headdim = v.shape
        with _on_device(v):
            device = v.device
            dout = dout.float().contiguous()
            v_k = _as_kernel_values(v)
            dv = torch.empty(batch, nheads, seqlen, headdim, device=device, dtype=torch.float32)
            dv_sink = torch.empty(
                ncomponents,
                batch * nheads,
                triton.cdiv(seqlen, BWD_PREPROCESS_BLOCK_M),
                headdim,
                device=device,
                dtype=torch.float32,
            )
            dgate_weights = torch.empty(batch, nheads, seqlen, ncomponents, device=device, dtype=torch.float32)
            grads = []
            for m in range(ncomponents):
                f, g, b = flat[3 * m : 3 * m + 3]
                df, dg, db = _order_attn_backward_impl(
                    dout,
                    f.float().contiguous(),
                    g.float().contiguous(),
                    v_k,
                    component_outs[m],
                    *saved[5 * m : 5 * m + 5],
                    chunk,
                    dv,
                    dv_sink[m],
                    None,
                    accumulate=m > 0,
                    gate_weights=gate_weights,
                    component=m,
                    dgate_weights=dgate_weights,
                )
                grads += [df.to(f.dtype), dg.to(g.dtype), db.to(b.dtype)]
            dv[:, :, 0] = dv_sink.sum((0, 2)).view(batch, nheads, headdim)  # key 0 is only ever the sink
            dgates = gate_weights * (dgate_weights - (gate_weights * dgate_weights).sum(-1, keepdim=True))  # softmax
        return (dgates.to(ctx.gates_dtype), dv.to(v.dtype), None, *grads)


def mixture_attn_func(
    gates: torch.Tensor, components: Sequence[Component], v: torch.Tensor, chunk: int = 64
) -> torch.Tensor:
    """Query-gated mixture of causal order heads: sum_m softmax(gates)_m order_attn_func(*components[m], v).

    One fused autograd function (``MixtureAttnFunc``) rather than autograd over the per-component outputs; when no
    gradient is needed (``torch.no_grad``, or no input requires grad) the forward keeps only one component's tensors
    alive at a time.

    Arguments:
        gates: (batch, nheads, seqlen, ncomponents) mixture logits.
        components: ncomponents tuples (f, g, b) as in ``order_attn_func``.
        v: (batch, nheads, seqlen, headdim) values shared by the components.
        chunk: see ``order_attn_func``.
    Return:
        out: (batch, nheads, seqlen, headdim) fp32 mixture output, differentiable in the gates, every component and v.
    """
    components = [tuple(c) for c in components]
    _check_mixture_inputs(gates, components, v, chunk)
    flat = [t for c in components for t in c]
    if torch.is_grad_enabled() and any(t.requires_grad for t in (gates, v, *flat)):
        return MixtureAttnFunc.apply(gates, v, chunk, *flat)
    with _on_device(v):
        return _mixture_attn_forward(gates, components, v, chunk, keep=False)[0]


mixture_attention_triton = mixture_attn_func  # backward-compatible name
