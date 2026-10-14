"""Fenwick-tree decode cache for causal K = 2 order attention with quantised thresholds.

A query x attends to the past keys y with logit min(g1(y) - f1(x), g2(y) - f2(x)) (see ``learning/order_attention``):
branch 1 (weight exp(g1 - f1)) when the key rank r(y) = g1(y) - g2(y) is <= the query threshold a(x) = f1(x) - f2(x),
branch 2 (weight exp(g2 - f2)) otherwise. Quantising ranks and thresholds to B cells (branch 1 iff cell(y) <= cut(x))
makes the state of a sequence independent of its length: per cell, one online-softmax state of exp(g1) [v, 1] and one
of exp(g2) [v, 1] over the keys that fell into it. A query needs the PREFIX of the branch-1 cell states over the cells
<= cut(x) and the SUFFIX of the branch-2 cell states over the cells > cut(x): two Fenwick trees (prefix sums under the
online-softmax combine; the second over reversed cell indices) answer both in O(log B) combines, and a new key enters
both in O(log B) combines. No sort and no growth with N: 2 (B + 2) (d + 2) fp32 words per head (B = 4096, d = 128:
4.3 MiB per head and sequence, the bf16 K + V of ~8.5k tokens of one head).

Only the branch decision is quantised, the logits are exact: a key whose rank falls into the same cell as the threshold
is scored on branch 1 even if its rank is slightly above the threshold. With one cell per representable value of the
realizers' numeric format (``bf16_edges``) the cache is exact for the bf16 model; on trained heads the branch flips are
harmless down to a few hundred cells (0.6B order students, 2026-10-12: +0.07813 nats to the base model unquantised,
+0.07810 at 4096 cells, +0.0784 at 256).

Pseudo code (one head; a state is (m, S) with S = sum exp(logit - m) [v, 1]; combine rescales two states to the larger
m and adds; shift((m, S), -f) = (m - f, S); normalise((m, S)) = S[:d] / S[d]; the exact algorithms are in the header
of ``learning/order_attention``):

    quantise:  edges e_1 < ... < e_{B-1};  cell(v) = #{k : e_k <= v} in 0 .. B-1
               branch 1 iff cell(r(y)) <= cell(a(x))                            # exact logits, quantised comparison

    per-cell states (what the trees aggregate):
        L1[c] = combine over keys y with cell(r(y)) = c of (g1(y), [v(y), 1])
        L2[c] = combine over keys y with cell(r(y)) = c of (g2(y), [v(y), 1])
        query x with cut = cell(a(x)) needs  combine(L1[0 .. cut])  and  combine(L2[cut+1 .. B-1])

    Fenwick trees T1 (cells in order; node i covers cells i - lowbit(i) .. i - 1) and T2 (cells reversed: c -> B - c):
        prefix(T, i):    acc = identity;  while i > 0:  acc = combine(acc, T[i]);  i -= lowbit(i)
        update(T, i, s):                  while i <= B: T[i] = combine(T[i], s);   i += lowbit(i)

    build(keys of a prompt):                                                      # O(N + B), no sort
        L1, L2 by scatter: m[c] = max logw over the cell, S[c] = sum exp(logw - m[c]) [v, 1]
        T1[c + 1] = L1[c];  T2[B - c] = L2[c]
        for i = 1 .. B:  j = i + lowbit(i);  if j <= B:  T[j] = combine(T[j], T[i])   # both trees

    step(x):                                                                      # O(log B) per token
        c = cell(r(x));  update(T1, c + 1, (g1(x), [v(x), 1]));  update(T2, B - c, (g2(x), [v(x), 1]))
        cut = cell(a(x))
        S1 = shift(prefix(T1, cut + 1), -f1(x))                                   # cells <= cut
        S2 = shift(prefix(T2, B - 1 - cut), -f2(x))                               # cells > cut
        out(x) = normalise(combine(S1, S2, (b(x), [v(0), 1])))

    training (``quantised_order_attn_func``, all queries of a sequence at once, with gradients):
        branch keys rank(y) = cell(r(y)), threshold(x) = cell(a(x)) are passed to the exact prefill kernels of
        ``learning/order_attention_triton`` (chunked position tree, sorted by the cell index instead of the rank),
        whose forward and backward then score the quantised model: d(out)/d(f, g, b, v) flow through the logits of the
        selected branches exactly as in the exact model, the cells carry no gradient. The decode cache above reproduces
        this forward token by token.

    max semiring (``semiring="max"``): node = (best logit, [v, position]) of its cells, combine = keep the larger
        logit (ties: the later key), the output is the winner's value and ``last_argmax`` its position (0 = sink).
        No exp and no normaliser: integer codes of any magnitude compare exactly.

Against the exact decode cache (``OrderCache``: rank-sorted blocks of sizes 2^l, a binary search per block, O(N d)
memory) the quantised cache replaces the sort by cell indexing and the per-block searches by fixed-length index
chains: 4 chains of <= log2(B) + 1 combines per token, 2 (B + 2)(d + 2) floats per head whatever N.

Measured (scripts/bench_fenwick_cache.py, GH200, 16 order heads, d = 128, fp32 states, median of 200 decode steps,
insert + query): the fused Triton step takes 0.034 / 0.044 / 0.054 ms at B = 256 / 4096 / 65536 for every context
length from 4k to 128k (query alone 0.018-0.027 ms; cache 4.1 / 65 / 1040 MiB; prefill build 9-44 ms at 128k).
``OrderCache`` (CUDA graph) steps in 0.086-0.094 ms with a p99 of 2.9 ms (binary-counter merges) and holds 123 MiB at
4k, 3.9 GB at 128k; a bf16 GQA-8 KV cache for the same 16 query heads steps in 0.039 / 0.057 / 0.128 / 0.222 ms at
4k / 16k / 64k / 128k with 16-512 MiB. The torch index chains are launch-bound (1.3-4.9 ms per step even from a CUDA
graph) and serve as the reference for the kernel.

Fenwick layout, per head, 1-indexed nodes 1..B: node i of tree 1 covers cells i - lowbit(i) .. i - 1 (0-indexed), node
i of tree 2 the same range of the REVERSED index B - 1 - cell. Node 0 holds the identity state (m = -inf, S = 0), so an
exhausted query chain reads "nothing"; node B + 1 is scratch and absorbs the writes of an exhausted update chain. A
prefix query walks i, i - lowbit(i), ...; an update walks i, i + lowbit(i), ... Every head follows its own chain, so
the chains run vectorised over heads as per-head gathers and index_puts, log2(B) + 1 steps each: a fixed number of
kernel launches for any N, optionally replayed from a CUDA graph.
"""
import logging
from typing import Callable, Optional

import torch

from learning.order_attention import (
    NEG, Semiring, _append_ones, _check_semiring, argmax_weights, online_softmax_combine,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the torch path works without Triton
    triton = None

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------------ quantisers
def uniform_edges(low: float, high: float, num_cells: int) -> torch.Tensor:
    """``num_cells - 1`` equally spaced cell edges over [low, high]."""
    return torch.linspace(low, high, num_cells - 1, dtype=torch.float32)


def quantile_edges(ranks: torch.Tensor, num_cells: int) -> torch.Tensor:
    """Per-head edges at the quantiles of observed ranks (num_heads, n) -> (num_heads, num_cells - 1): equal counts."""
    q = torch.linspace(0, 1, num_cells + 1, dtype=torch.float64, device=ranks.device)[1:-1]
    return torch.quantile(ranks.double(), q, dim=-1).transpose(0, 1).contiguous().float()


def bf16_edges(low: float, high: float) -> torch.Tensor:
    """Every finite bfloat16 value in [low, high], ascending: one cell per representable rank (floor to bf16)."""
    codes = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32).to(torch.int16)
    vals = codes.view(torch.bfloat16).float()
    vals = vals[torch.isfinite(vals) & (vals >= low) & (vals <= high)]
    return torch.unique(vals)  # sorted, -0.0 and 0.0 merged


def rank_cells(edges: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Cell index #{edges <= value} in [0, num_cells - 1]; edges (num_cells - 1,) or (num_heads, num_cells - 1) sorted,
    values (num_heads, n) (or any shape for 1-D edges). Monotone, so r <= a implies cell(r) <= cell(a)."""
    values = torch.nan_to_num(values.float(), nan=0.0).contiguous()
    if edges.dim() == 1:
        return torch.searchsorted(edges, values, right=True)
    return torch.searchsorted(edges, values.reshape(edges.shape[0], -1), right=True).reshape(values.shape)


# --------------------------------------------------------------------------------------------------------- states
Combine = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def select_combine(
    m_a: torch.Tensor, S_a: torch.Tensor, m_b: torch.Tensor, S_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max-semiring combine of (logit, payload [v, position]) states: the larger logit wins, ties go to the later
    position, a -inf state never wins. (-inf, 0) is the identity."""
    idx_a, idx_b = S_a[..., -1:], S_b[..., -1:]
    take_b = (m_b > m_a) | ((m_b == m_a) & (idx_b > idx_a) & (m_b > NEG))
    return torch.where(take_b, m_b, m_a), torch.where(take_b, S_b, S_a)


def _combine_of(semiring: str) -> Combine:
    return online_softmax_combine if semiring == "logsumexp" else select_combine


def _payload(value: torch.Tensor, positions: Optional[torch.Tensor], semiring: str) -> torch.Tensor:
    """[v, 1] (the normaliser channel) or [v, position] (the winner's identity) appended to the values (..., d)."""
    if semiring == "logsumexp":
        return _append_ones(value)
    return torch.cat([value, positions.to(value.dtype).expand(*value.shape[:-1]).unsqueeze(-1)], dim=-1)


def _cell_states(
    cells: torch.Tensor, logw: torch.Tensor, payload: torch.Tensor, num_cells: int, semiring: str = "logsumexp"
) -> torch.Tensor:
    """Per-cell state [S, m]: the online-softmax sum of the cell's keys (log-sum-exp), or its best key (max).

    Args:
        cells: (num_heads, n) cell of each key.
        logw: (num_heads, n) log weight of each key (-inf = masked).
        payload: (num_heads, n, width) augmented values [v, 1] or [v, position].
        num_cells: B.
        semiring: "logsumexp" or "max".

    Returns:
        (num_heads, B, width + 1); empty cells hold the identity (S = 0, m = -inf).
    """
    num_heads, _, width = payload.shape
    m = torch.full((num_heads, num_cells), NEG, device=payload.device)
    m = m.scatter_reduce(1, cells, logw, reduce="amax", include_self=True)
    m_at = m.gather(1, cells)
    if semiring == "logsumexp":
        w = torch.where(logw == NEG, 0.0, torch.exp(logw - torch.where(m_at == NEG, 0.0, m_at)))
        S = torch.zeros((num_heads, num_cells, width), device=payload.device)
        S.scatter_add_(1, cells.unsqueeze(-1).expand(-1, -1, width), w.unsqueeze(-1) * payload)
    else:  # the winner of a cell: among its maximal finite keys, the one with the largest position
        cand = torch.where((logw == m_at) & (logw > NEG), payload[..., -1], -1.0)
        best = torch.full((num_heads, num_cells), -1.0, device=payload.device)
        best = best.scatter_reduce(1, cells, cand, reduce="amax", include_self=True)
        order = torch.arange(payload.shape[1], device=payload.device).expand(num_heads, -1)
        winner = torch.full((num_heads, num_cells), -1, dtype=torch.long, device=payload.device)
        winner = winner.scatter_reduce(1, cells, torch.where(cand == best.gather(1, cells), order, -1), reduce="amax")
        rows = payload.gather(1, winner.clamp(min=0).unsqueeze(-1).expand(-1, -1, width))
        S = torch.where((winner >= 0).unsqueeze(-1), rows, 0.0)
    return torch.cat([S, m.unsqueeze(-1)], dim=-1)


def _fenwick_build_(tree: torch.Tensor, num_cells: int, combine: Combine = online_softmax_combine) -> None:
    """In place, from leaves at nodes 1..B (one per cell): node i += node j for every j with j + lowbit(j) = i.

    Processed by lowbit level (1, 2, 4, ...): a node receives only from lower levels and sends only to higher ones, so
    every level is one vectorised gather / combine / scatter.
    """
    for level in range(num_cells.bit_length()):
        step = 1 << level
        stop = num_cells - step + 1  # sources i = step, 3 step, ... with i + step <= B
        if stop <= step:
            break
        src = torch.arange(step, stop, 2 * step, device=tree.device)
        dst = src + step
        m, S = combine(tree[:, dst, -1:], tree[:, dst, :-1], tree[:, src, -1:], tree[:, src, :-1])
        tree[:, dst] = torch.cat([S, m], dim=-1)


class OrderFenwickCache:
    """Decode cache of one K = 2 order head group (one sequence, H heads) with ranks quantised to B cells.

    Args:
        edges: (B - 1,) cell edges shared by the heads, or (H, B - 1) per head; sorted ascending (see ``uniform_edges``,
            ``quantile_edges``, ``bf16_edges``). Cell(v) = #{edges <= v}.
        cuda_graph: replay ``step`` and ``query`` from CUDA graphs captured lazily (CUDA only; the torch path).
        impl: "torch" (vectorised index chains, log B kernel launches per chain) or "triton" (one fused launch per
            step: cell search, both updates, both prefix chains, sink and normalisation; CUDA only).
        semiring: "logsumexp" (softmax-weighted mean; node rows [S, m] with S = sum exp(g - m) [v, 1]) or "max"
            (the best key; rows [v, position, m]; ``last_argmax`` holds the winners' positions after each query).
    """

    def __init__(
        self, edges: torch.Tensor, cuda_graph: bool = False, impl: str = "torch", semiring: Semiring = "logsumexp"
    ):
        edges = edges.float()
        if impl not in ("torch", "triton"):
            raise ValueError(f"impl must be 'torch' or 'triton', got {impl!r}")
        if impl == "triton" and triton is None:
            raise RuntimeError("impl='triton' needs Triton")
        _check_semiring(semiring)
        self.impl = impl
        self.semiring = semiring
        self.combine = _combine_of(semiring)
        if edges.dim() not in (1, 2) or edges.shape[-1] < 1:
            raise ValueError(f"edges must be (B - 1,) or (H, B - 1) with B >= 2, got {tuple(edges.shape)}")
        if not bool((edges[..., 1:] >= edges[..., :-1]).all()):
            raise ValueError("edges must be sorted ascending")
        self.edges = edges
        self.cuda_graph = cuda_graph
        self.num_cells = edges.shape[-1] + 1
        self.num_steps = self.num_cells.bit_length()  # longest query or update chain
        self.v0: Optional[torch.Tensor] = None  # (H, d + 1) sink value
        self.tree1: Optional[torch.Tensor] = None  # (H, B + 2, d + 2): [S (d + 1), m], branch-1 prefix tree
        self.tree2: Optional[torch.Tensor] = None  # branch-2 tree over reversed cells
        self.n = 0  # tokens seen, sink included
        self.captures = 0
        self._graphs: dict[tuple, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self._pool = None

    # ----------------------------------------------------------------------------------------------------- layout
    def cells(self, values: torch.Tensor) -> torch.Tensor:
        """Cell index of ranks or thresholds (H, n) -> (H, n) int64."""
        return rank_cells(self.edges, values)

    @property
    def last_argmax(self) -> Optional[torch.Tensor]:
        """(H,) int64 positions of the winners of the last step or query (max semiring; 0 = the sink)."""
        return None if self.tree1 is None else self._out_idx.long()

    def nbytes(self) -> int:
        """Bytes of the two trees (independent of the number of tokens)."""
        return 0 if self.tree1 is None else 2 * self.tree1.numel() * 4

    def _alloc(self, num_heads: int, head_dim: int, device: torch.device) -> None:
        if self.edges.dim() == 2 and self.edges.shape[0] != num_heads:
            raise ValueError(f"edges are for {self.edges.shape[0]} heads, values have {num_heads}")
        self.edges = self.edges.to(device)
        shape = (num_heads, self.num_cells + 2, head_dim + 2)
        self.tree1 = torch.zeros(shape, device=device)
        self.tree1[..., -1] = NEG
        self.tree2 = self.tree1.clone()
        self._heads = torch.arange(num_heads, device=device)
        self.edges = self.edges.contiguous()
        self._out = torch.empty((num_heads, head_dim), device=device)
        self._out_idx = torch.empty((num_heads,), device=device)
        self._next_pos = torch.zeros((), device=device)  # position of the next key, kept on the device (no copies)
        self._graphs.clear()
        if self.cuda_graph and device.type == "cuda":
            self._f_in = torch.zeros((num_heads, 2), device=device)
            self._g_in = torch.zeros((num_heads, 2), device=device)
            self._b_in = torch.zeros((num_heads,), device=device)
            self._v_in = torch.zeros((num_heads, head_dim + 1), device=device)

    # ---------------------------------------------------------------------------------------------------- prefill
    @torch.no_grad()
    def prefill(self, g: torch.Tensor, value: torch.Tensor) -> None:
        """Resets the cache to a prompt (its attention outputs come from the prefill kernel, not from here).

        Args:
            g: (H, N, 2) key realizers of positions 0..N-1 (0 = sink).
            value: (H, N, d) values.
        """
        g, value = g.float(), value.float()
        num_heads, seq_len, head_dim = value.shape
        positions = torch.arange(seq_len, device=value.device)
        value = _payload(value, positions, self.semiring)
        self._alloc(num_heads, head_dim, value.device)
        self.v0 = value[:, 0].contiguous()  # the kernel reads it as (H, d + 1) contiguous
        self.n = seq_len
        self._next_pos.fill_(seq_len)
        if seq_len > 1:
            self._build(g[:, 1:], value[:, 1:])

    def _build(self, g: torch.Tensor, value: torch.Tensor) -> None:
        """Leaves from the per-cell states of the keys, then one O(B) bottom-up pass per tree."""
        B = self.num_cells
        cells = self.cells(g[..., 0] - g[..., 1])
        self.tree1[:, 1 : B + 1] = _cell_states(cells, g[..., 0], value, B, self.semiring)  # node c + 1 = cell c
        self.tree2[:, 1 : B + 1] = _cell_states(cells, g[..., 1], value, B, self.semiring).flip(1)  # node B - c
        _fenwick_build_(self.tree1, B, self.combine)
        _fenwick_build_(self.tree2, B, self.combine)

    # ------------------------------------------------------------------------------------------------ chains
    def _prefix(self, tree: torch.Tensor, start: torch.Tensor, shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine of the nodes start, start - lowbit(start), ... (per head), logits shifted by -shift (H, 1)."""
        num_heads, width = start.shape[0], tree.shape[-1] - 1
        m = torch.full((num_heads, 1), NEG, device=tree.device)
        S = torch.zeros((num_heads, width), device=tree.device)
        idx = start
        for _ in range(self.num_steps):
            node = tree[self._heads, idx]  # node 0 is the identity: an exhausted chain adds nothing
            m, S = self.combine(m, S, node[:, -1:] - shift, node[:, :-1])
            idx = idx - (idx & -idx)
        return m, S

    def _update(self, tree: torch.Tensor, start: torch.Tensor, m_key: torch.Tensor, S_key: torch.Tensor) -> None:
        """Combine one key state (m (H, 1), S (H, width)) into the nodes start, start + lowbit(start), ... <= B."""
        idx = start
        for _ in range(self.num_steps):
            node = tree[self._heads, idx]
            m, S = self.combine(node[:, -1:], node[:, :-1], m_key, S_key)
            tree[self._heads, idx] = torch.cat([S, m], dim=-1)
            idx = idx + (idx & -idx)
            idx = torch.where(idx > self.num_cells, self.num_cells + 1, idx)  # exhausted: write the scratch node

    def _query_impl(self, f: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        cut = self.cells(f[:, 0] - f[:, 1])  # (H,)
        m1, S1 = self._prefix(self.tree1, cut + 1, f[:, :1])  # cells <= cut
        m2, S2 = self._prefix(self.tree2, self.num_cells - 1 - cut, f[:, 1:])  # cells > cut (reversed index)
        m, S = self.combine(m1, S1, m2, S2)
        m, S = self.combine(m, S, b.unsqueeze(-1), self.v0)
        if self.semiring == "max":
            self._out_idx.copy_(S[:, -1])  # in place: the same buffer under eager and CUDA-graph execution
            return S[:, :-1].clone()
        return S[:, :-1] / S[:, -1:]

    def _step_impl(self, f: torch.Tensor, g: torch.Tensor, b: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        cell = self.cells(g[:, 0] - g[:, 1])  # (H,)
        self._update(self.tree1, cell + 1, g[:, :1], v)
        self._update(self.tree2, self.num_cells - cell, g[:, 1:], v)
        return self._query_impl(f, b)

    # ------------------------------------------------------------------------------------------------ graphs
    def _replay(self, key: tuple, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        """Run ``fn`` from a CUDA graph captured on first use (one shared pool; callers clone the static output)."""
        if key not in self._graphs:
            logger.debug("OrderFenwickCache: capturing CUDA graph %s", key)
            snapshot = (self.tree1.clone(), self.tree2.clone())  # warm-up and capture must not mutate the trees
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fn()
            torch.cuda.current_stream().wait_stream(stream)
            self.tree1.copy_(snapshot[0])
            self.tree2.copy_(snapshot[1])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                out = fn()
            self._pool = graph.pool()
            self.tree1.copy_(snapshot[0])
            self.tree2.copy_(snapshot[1])
            self._graphs[key] = (graph, out)
            self.captures += 1
        graph, out = self._graphs[key]
        graph.replay()
        return out

    def _use_graphs(self, x: torch.Tensor) -> bool:
        return self.cuda_graph and x.is_cuda

    # ------------------------------------------------------------------------------------------------ decoding
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
        if self.v0 is None:  # the first token is the sink
            self.prefill(g.unsqueeze(1), v.unsqueeze(1))
            self._out_idx.zero_()
            return v.clone()
        v = _payload(v, self._next_pos, self.semiring)  # the new key's position is n
        if self.semiring == "max":
            self._next_pos += 1
        self.n += 1
        if self.impl == "triton" and f.is_cuda:
            return self._launch(f, g, b, v, update=True)
        if self._use_graphs(f):
            torch._foreach_copy_([self._f_in, self._g_in, self._b_in, self._v_in], [f, g, b, v])
            return self._replay(("step",), lambda: self._step_impl(self._f_in, self._g_in, self._b_in, self._v_in)).clone()
        return self._step_impl(f, g, b, v)

    @torch.no_grad()
    def query(self, f: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Attention output (H, d) of query realizers f (H, 2) and sink logit b (H,) over the cached keys."""
        f, b = f.float(), b.float()
        if self.impl == "triton" and f.is_cuda:
            return self._launch(f, b.new_zeros((f.shape[0], 2)), b, self.v0, update=False)
        if self._use_graphs(f):
            torch._foreach_copy_([self._f_in, self._b_in], [f, b])
            return self._replay(("query",), lambda: self._query_impl(self._f_in, self._b_in)).clone()
        return self._query_impl(f, b)

    def _launch(self, f: torch.Tensor, g: torch.Tensor, b: torch.Tensor, v: torch.Tensor, update: bool) -> torch.Tensor:
        """One fused kernel launch: (optionally) insert the key g / v, then answer the query f / b. v is augmented."""
        num_heads, width = self.tree1.shape[0], self.tree1.shape[-1]
        num_edges = self.edges.shape[-1]
        _fenwick_step_kernel[(num_heads,)](
            self.tree1, self.tree2, self.edges, f.contiguous(), g.contiguous(), b.contiguous(), v.contiguous(),
            self.v0, self._out, self._out_idx, self.num_cells, num_edges,
            self.edges.stride(0) if self.edges.dim() == 2 else 0, WIDTH=width, BLOCK_D=triton.next_power_of_2(width),
            NUM_STEPS=self.num_steps, NUM_SEARCH=(num_edges + 1).bit_length(), DO_UPDATE=update,
            SEMIRING_MAX=self.semiring == "max", num_warps=max(1, min(4, width // 64)),
        )
        return self._out.clone()


# --------------------------------------------------------------------------------------------------------- kernel
if triton is not None:

    @triton.jit
    def _fenwick_combine(m_acc, S_acc, m_new, S_new, cols, WIDTH: tl.constexpr, SEMIRING_MAX: tl.constexpr):
        """Combine of two states; m are (1,) tensors, S are (BLOCK_D,) rows [payload, (m column zeroed)].

        Log-sum-exp: online-softmax rescale and add ((-inf, 0) is the identity). Max: keep the state with the larger
        logit, ties to the larger position (payload column WIDTH - 2), a -inf state never wins.
        """
        if SEMIRING_MAX:
            idx_acc = tl.sum(tl.where(cols == WIDTH - 2, S_acc, 0.0), axis=0)
            idx_new = tl.sum(tl.where(cols == WIDTH - 2, S_new, 0.0), axis=0)
            take = (m_new > m_acc) | ((m_new == m_acc) & (idx_new > idx_acc) & (m_new > float("-inf")))
            return tl.where(take, m_new, m_acc), tl.where(take, S_new, S_acc)
        m = tl.maximum(m_acc, m_new)
        m_s = tl.where(m == float("-inf"), 0.0, m)
        w_acc = tl.where(m_acc == float("-inf"), 0.0, tl.exp(tl.where(m_acc == float("-inf"), 0.0, m_acc) - m_s))
        w_new = tl.where(m_new == float("-inf"), 0.0, tl.exp(tl.where(m_new == float("-inf"), 0.0, m_new) - m_s))
        return m, S_acc * w_acc + S_new * w_new

    @triton.jit
    def _fenwick_search(Edges, num_edges, value, NUM_SEARCH: tl.constexpr):
        """#{edges <= value} by a branchless binary search; value is a (1,) tensor."""
        lo = tl.zeros((1,), dtype=tl.int32)
        hi = lo + num_edges
        for _ in range(NUM_SEARCH):
            active = lo < hi
            mid = (lo + hi) // 2
            e = tl.load(Edges + tl.minimum(mid, num_edges - 1))
            le = e <= value
            lo = tl.where(active & le, mid + 1, lo)
            hi = tl.where(active & (le == 0), mid, hi)
        return lo

    @triton.jit
    def _fenwick_prefix(Tree, start, shift, cols, mask, WIDTH: tl.constexpr, NUM_STEPS: tl.constexpr,
                        SEMIRING_MAX: tl.constexpr):
        """Combine of the nodes start, start - lowbit(start), ... (node 0 = identity), logits shifted by -shift."""
        m = tl.full((1,), float("-inf"), dtype=tl.float32)
        S = tl.zeros_like(cols.to(tl.float32))
        idx = start
        for _ in range(NUM_STEPS):
            row = Tree + idx * WIDTH
            node = tl.load(row + cols, mask=mask, other=0.0)
            m_node = tl.load(row + (WIDTH - 1) + tl.zeros((1,), dtype=tl.int32))
            S_node = tl.where(cols == WIDTH - 1, 0.0, node)
            m, S = _fenwick_combine(m, S, m_node - shift, S_node, cols, WIDTH, SEMIRING_MAX)
            idx = idx - (idx & (-idx))
        return m, S

    @triton.jit
    def _fenwick_update(Tree, start, m_key, S_key, num_cells, cols, mask, WIDTH: tl.constexpr,
                        NUM_STEPS: tl.constexpr, SEMIRING_MAX: tl.constexpr):
        """Combine the key state into the nodes start, start + lowbit(start), ... <= B (then the scratch node)."""
        idx = start
        for _ in range(NUM_STEPS):
            row = Tree + idx * WIDTH
            node = tl.load(row + cols, mask=mask, other=0.0)
            m_node = tl.load(row + (WIDTH - 1) + tl.zeros((1,), dtype=tl.int32))
            S_node = tl.where(cols == WIDTH - 1, 0.0, node)
            m, S = _fenwick_combine(m_node, S_node, m_key, S_key, cols, WIDTH, SEMIRING_MAX)
            tl.store(row + cols, tl.where(cols == WIDTH - 1, m, S), mask=mask)
            idx = idx + (idx & (-idx))
            idx = tl.where(idx > num_cells, num_cells + 1, idx)

    @triton.jit
    def _fenwick_step_kernel(T1, T2, Edges, F, G, Bsink, Vaug, V0, Out, OutIdx, num_cells, num_edges, stride_edges,
                             WIDTH: tl.constexpr, BLOCK_D: tl.constexpr, NUM_STEPS: tl.constexpr,
                             NUM_SEARCH: tl.constexpr, DO_UPDATE: tl.constexpr, SEMIRING_MAX: tl.constexpr):
        """One program per head: insert the key (DO_UPDATE), then the query over both trees and the sink.

        T1, T2 (H, B + 2, WIDTH) trees with rows [payload (d + 1), m]; Edges (B - 1,) or (H, B - 1) with row stride
        ``stride_edges`` (0 = shared); F, G (H, 2); Bsink (H,); Vaug (H, d + 1) the augmented value of the new key
        ([v, 1], or [v, position] in the max semiring); V0 (H, d + 1) the sink's; Out (H, d) the attention output;
        OutIdx (H,) the winner's position (max semiring; -1 otherwise).
        """
        h = tl.program_id(0)
        one = tl.zeros((1,), dtype=tl.int32)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < WIDTH
        smask = cols < WIDTH - 1
        f1 = tl.load(F + h * 2 + one)
        f2 = tl.load(F + h * 2 + 1 + one)
        edges = Edges + h * stride_edges
        rows = num_cells + 2
        tree1 = T1 + h * rows * WIDTH
        tree2 = T2 + h * rows * WIDTH
        if DO_UPDATE:
            g1 = tl.load(G + h * 2 + one)
            g2 = tl.load(G + h * 2 + 1 + one)
            key_S = tl.load(Vaug + h * (WIDTH - 1) + cols, mask=smask, other=0.0)
            cell = _fenwick_search(edges, num_edges, g1 - g2, NUM_SEARCH)
            _fenwick_update(tree1, cell + 1, g1, key_S, num_cells, cols, mask, WIDTH, NUM_STEPS, SEMIRING_MAX)
            _fenwick_update(tree2, num_cells - cell, g2, key_S, num_cells, cols, mask, WIDTH, NUM_STEPS, SEMIRING_MAX)
            tl.debug_barrier()
        cut = _fenwick_search(edges, num_edges, f1 - f2, NUM_SEARCH)
        m, S = _fenwick_prefix(tree1, cut + 1, f1, cols, mask, WIDTH, NUM_STEPS, SEMIRING_MAX)
        m2, S2 = _fenwick_prefix(tree2, num_cells - 1 - cut, f2, cols, mask, WIDTH, NUM_STEPS, SEMIRING_MAX)
        m, S = _fenwick_combine(m, S, m2, S2, cols, WIDTH, SEMIRING_MAX)
        v0 = tl.load(V0 + h * (WIDTH - 1) + cols, mask=smask, other=0.0)
        m, S = _fenwick_combine(m, S, tl.load(Bsink + h + one), v0, cols, WIDTH, SEMIRING_MAX)
        last = tl.sum(tl.where(cols == WIDTH - 2, S, 0.0), axis=0)  # the normaliser, or the winner's position
        if SEMIRING_MAX:
            tl.store(Out + h * (WIDTH - 2) + cols, S, mask=cols < WIDTH - 2)
            tl.store(OutIdx + h + one, last + tl.zeros((1,), dtype=tl.float32))
        else:
            tl.store(Out + h * (WIDTH - 2) + cols, S / last, mask=cols < WIDTH - 2)
            tl.store(OutIdx + h + one, tl.full((1,), -1.0, dtype=tl.float32))


# ------------------------------------------------------------------------------------------------------ training
def _cells_batched(edges: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Cell indices of (batch, num_heads, seq_len) values as fp32 (edges shared or per head)."""
    if edges.dim() == 1:
        return rank_cells(edges, values).float()
    batch, num_heads, seq_len = values.shape
    cells = rank_cells(edges, values.transpose(0, 1).reshape(num_heads, batch * seq_len))
    return cells.reshape(num_heads, batch, seq_len).transpose(0, 1).float()


def quantised_order_attn_func(
    f: torch.Tensor, g: torch.Tensor, b: torch.Tensor, v: torch.Tensor, edges: torch.Tensor, chunk: int = 64
) -> torch.Tensor:
    """Causal order attention of the quantised model, for training: the Triton forward and backward of
    ``order_attn_func`` with the branch keys replaced by the cell indices of ranks and thresholds, so that a model
    trained with it decodes EXACTLY through ``OrderFenwickCache`` with the same edges. Differentiable in f, g, b and
    v; the cells are piecewise constant and carry no gradient (as the branch decision of the exact model).

    Arguments:
        f: (batch, nheads, seqlen, 2) query realizers.
        g: (batch, nheads, seqlen, 2) key realizers (position 0 is the sink key).
        b: (batch, nheads, seqlen) sink logits.
        v: (batch, nheads, seqlen, headdim) values.
        edges: (B - 1,) shared or (nheads, B - 1) per-head cell edges, as in ``OrderFenwickCache``.
        chunk: see ``order_attn_func``.
    Return:
        (batch, nheads, seqlen, headdim) fp32 attention output.
    """
    from learning.order_attention_triton import order_attn_func

    edges = edges.to(f.device).float()
    with torch.no_grad():
        krank = _cells_batched(edges, g[..., 0].float() - g[..., 1].float())
        qthr = _cells_batched(edges, f[..., 0].float() - f[..., 1].float())
    return order_attn_func(f, g, b, v, chunk, rank=krank, threshold=qthr)


# ------------------------------------------------------------------------------------------------------ reference
def eager_quantised_order_attention_forward(
    f: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    value: torch.Tensor,
    edges: torch.Tensor,
    semiring: Semiring = "logsumexp",
) -> tuple[torch.Tensor, torch.Tensor]:
    """O(N^2) fp64 reference of the quantised model: branch by cells, exact logits, causal, sink at key 0.

    Args:
        f: (H, N, 2) query realizers.
        g: (H, N, 2) key realizers.
        b: (H, N) sink logits.
        value: (H, N, d) values.
        edges: cell edges as in ``OrderFenwickCache``.
        semiring: "logsumexp" (softmax) or "max" (one-hot on the best key, ties to the later key).

    Returns:
        attn_output (H, N, d) and attn_weights (H, N, N), both float64.
    """
    _check_semiring(semiring)
    cell = rank_cells(edges, g[..., 0] - g[..., 1])  # (H, N)
    cut = rank_cells(edges, f[..., 0] - f[..., 1])
    f, g, b, value = f.double(), g.double(), b.double(), value.double()
    d1 = g[:, None, :, 0] - f[:, :, None, 0]  # (H, x, y)
    d2 = g[:, None, :, 1] - f[:, :, None, 1]
    scores = torch.where(cell[:, None, :] <= cut[:, :, None], d1, d2)
    scores = torch.cat([b.unsqueeze(-1), scores[..., 1:]], dim=-1)
    seq_len = scores.shape[-1]
    scores = scores.masked_fill(~torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device).tril(), NEG)
    attn_weights = torch.softmax(scores, dim=-1) if semiring == "logsumexp" else argmax_weights(scores)
    return attn_weights @ value, attn_weights
