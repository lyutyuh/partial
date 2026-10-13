"""Tests of the Fenwick-tree decode cache with quantised thresholds (learning/fenwick_cache.py) against the dense
quantised reference, the exact dense reference (when every distinct rank is its own cell) and ``OrderCache``.

Run from the repo root: python -m pytest tests/test_fenwick_cache.py -q
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.fenwick_cache import (  # noqa: E402
    OrderFenwickCache, bf16_edges, eager_quantised_order_attention_forward, quantile_edges, rank_cells, uniform_edges,
)
from learning.order_attention import OrderCache, dense_order_attention  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _inputs(seed, H, N, d, scale=2.0):
    gen = torch.Generator().manual_seed(seed)
    f = (torch.randn(H, N, 2, generator=gen) * scale).to(DEVICE)
    g = (torch.randn(H, N, 2, generator=gen) * scale).to(DEVICE)
    b = (torch.randn(H, N, generator=gen) * scale).to(DEVICE)
    V = torch.randn(H, N, d, generator=gen).to(DEVICE)
    return f, g, b, V


def _edges(kind, f, g, num_cells):
    """Cell edges of a test kind; 'ties' also snaps half of the ranks and thresholds exactly onto edges."""
    rank, thr = g[..., 0] - g[..., 1], f[..., 0] - f[..., 1]
    if kind == "uniform":
        return uniform_edges(-6.0, 6.0, num_cells).to(DEVICE)
    if kind == "quantile":
        return quantile_edges(rank, num_cells)
    if kind == "exact":  # every distinct rank and threshold is its own cell: the quantised model is the exact one
        return torch.unique(torch.cat([rank.flatten(), thr.flatten()]))
    if kind == "ties":
        edges = quantile_edges(rank, num_cells)
        H, N = rank.shape
        sel = torch.arange(N, device=DEVICE) % 2 == 0
        near = (rank[..., None] - edges[:, None]).abs().argmin(-1)  # (H, N) nearest edge per key
        g[..., 1] = torch.where(sel, g[..., 0] - edges.gather(1, near), g[..., 1])
        near = (thr[..., None] - edges[:, None]).abs().argmin(-1)
        f[..., 1] = torch.where(~sel, f[..., 0] - edges.gather(1, near), f[..., 1])
        return edges
    raise ValueError(kind)


def _check_decode(cache, f, g, b, V, ref, prefill, rtol=1e-4, atol=1e-5):
    N = f.shape[1]
    if prefill:
        cache.prefill(g[:, :prefill], V[:, :prefill])
    for t in range(prefill, N):
        out = cache.step(f[:, t], g[:, t], b[:, t], V[:, t])
        torch.testing.assert_close(out, ref[:, t].float(), rtol=rtol, atol=atol)
        torch.testing.assert_close(cache.query(f[:, t], b[:, t]), out, rtol=0, atol=0)
    assert cache.n == N


@pytest.mark.parametrize("kind", ["uniform", "quantile", "ties"])
@pytest.mark.parametrize("H,N,d,num_cells,prefill", [(2, 1, 4, 2, 0), (2, 9, 4, 3, 0), (3, 40, 8, 16, 0),
                                                     (3, 40, 8, 5, 13), (2, 70, 5, 64, 64), (2, 70, 5, 1000, 70),
                                                     (4, 130, 6, 7, 1)])
def test_decoding_matches_quantised_reference(kind, H, N, d, num_cells, prefill):
    """Prefill, then decode token by token: every output equals the dense reference with quantised branches."""
    f, g, b, V = _inputs(N * 7 + prefill + len(kind), H, N, d)
    edges = _edges(kind, f, g, num_cells)
    ref = eager_quantised_order_attention_forward(f, g, b, V, edges)[0]
    _check_decode(OrderFenwickCache(edges), f, g, b, V, ref, prefill)


@pytest.mark.parametrize("H,N,d,prefill", [(2, 9, 4, 0), (3, 40, 8, 13), (2, 70, 5, 64)])
def test_exact_edges_match_dense_and_order_cache(H, N, d, prefill):
    """With one cell per distinct value the cache equals the unquantised dense reference and ``OrderCache``."""
    f, g, b, V = _inputs(N * 3 + prefill, H, N, d)
    edges = _edges("exact", f, g, 0)
    ref = dense_order_attention(f[None], g[None], b[None], V[None], True)[0]
    _check_decode(OrderFenwickCache(edges), f, g, b, V, ref, prefill)
    cache, other = OrderFenwickCache(edges), OrderCache(chunk=8)
    cache.prefill(g[:, :prefill], V[:, :prefill]) if prefill else None
    other.prefill(g[:, :prefill], V[:, :prefill]) if prefill else None
    for t in range(prefill, N):
        a = cache.step(f[:, t], g[:, t], b[:, t], V[:, t])
        torch.testing.assert_close(a, other.step(f[:, t], g[:, t], b[:, t], V[:, t]), rtol=1e-5, atol=1e-5)


def test_masked_keys_and_sink_only():
    """Keys with g = -inf contribute nothing; with every key masked the output is the sink value."""
    H, N, d = 2, 33, 4
    f, g, b, V = _inputs(5, H, N, d)
    g[:, 3:9] = float("-inf")
    g[:, 20] = float("-inf")
    edges = uniform_edges(-6.0, 6.0, 8).to(DEVICE)
    ref = eager_quantised_order_attention_forward(f, g, b, V, edges)[0]
    _check_decode(OrderFenwickCache(edges), f, g, b, V, ref, 10)
    g[:, 1:] = float("-inf")
    cache = OrderFenwickCache(edges)
    cache.prefill(g, V)
    torch.testing.assert_close(cache.query(f[:, -1], b[:, -1]), V[:, 0], rtol=0, atol=0)


def test_bf16_edges_cache_and_small_deviation():
    """One cell per bfloat16 value: equals the quantised reference, and stays close to the unquantised model."""
    H, N, d = 2, 60, 4
    f, g, b, V = _inputs(11, H, N, d, scale=1.0)
    edges = bf16_edges(-16.0, 16.0).to(DEVICE)
    assert edges.numel() > 2000 and bool((edges[1:] > edges[:-1]).all())
    ref = eager_quantised_order_attention_forward(f, g, b, V, edges)[0]
    _check_decode(OrderFenwickCache(edges), f, g, b, V, ref, 20)
    exact = dense_order_attention(f[None], g[None], b[None], V[None], True)[0]
    assert (ref.float() - exact).abs().max() < 0.1


def test_rank_cells_edges_and_monotone():
    edges = torch.tensor([-1.0, 0.0, 2.0])
    vals = torch.tensor([[-5.0, -1.0, -0.5, 0.0, 1.0, 2.0, 3.0, float("nan")]])
    assert rank_cells(edges, vals).tolist() == [[0, 1, 1, 2, 2, 3, 3, 2]]
    per_head = torch.stack([edges, edges + 1])
    assert rank_cells(per_head, vals.expand(2, -1).contiguous()).tolist() == [[0, 1, 1, 2, 2, 3, 3, 2],
                                                                              [0, 0, 0, 1, 2, 2, 3, 1]]
    with pytest.raises(ValueError):
        OrderFenwickCache(torch.tensor([1.0, 0.0]))


def test_nbytes_independent_of_length():
    H, d, B = 3, 5, 64
    edges = uniform_edges(-6.0, 6.0, B).to(DEVICE)
    sizes = []
    for N in (2, 50, 700):
        f, g, b, V = _inputs(N, H, N, d)
        cache = OrderFenwickCache(edges)
        cache.prefill(g, V)
        sizes.append(cache.nbytes())
    assert len(set(sizes)) == 1 and sizes[0] == 2 * H * (B + 2) * (d + 2) * 4


@pytest.mark.skipif(DEVICE != "cuda", reason="CUDA graphs")
@pytest.mark.parametrize("prefill", [0, 1, 37])
def test_cuda_graph_matches_eager(prefill):
    H, N, d = 3, prefill + 40, 5
    f, g, b, V = _inputs(100 + prefill, H, N, d)
    edges = quantile_edges(g[..., 0] - g[..., 1], 16)
    eager, graph = OrderFenwickCache(edges), OrderFenwickCache(edges, cuda_graph=True)
    if prefill:
        eager.prefill(g[:, :prefill], V[:, :prefill])
        graph.prefill(g[:, :prefill], V[:, :prefill])
    for t in range(prefill, N):
        a = eager.step(f[:, t], g[:, t], b[:, t], V[:, t])
        c = graph.step(f[:, t], g[:, t], b[:, t], V[:, t])
        torch.testing.assert_close(c, a, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(graph.query(f[:, t], b[:, t]), eager.query(f[:, t], b[:, t]), rtol=1e-6, atol=1e-6)
    assert graph.captures == 2
    torch.testing.assert_close(graph.tree1, eager.tree1, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(DEVICE != "cuda", reason="Triton kernel")
@pytest.mark.parametrize("per_head,prefill,num_cells", [(False, 0, 16), (True, 37, 16), (True, 5, 1000), (False, 64, 2)])
def test_triton_step_matches_torch(per_head, prefill, num_cells):
    H, N, d = 3, prefill + 50, 5
    f, g, b, V = _inputs(300 + prefill + num_cells, H, N, d)
    edges = quantile_edges(g[..., 0] - g[..., 1], num_cells) if per_head else uniform_edges(-6.0, 6.0, num_cells).cuda()
    ref = eager_quantised_order_attention_forward(f, g, b, V, edges)[0]
    torch_cache, triton_cache = OrderFenwickCache(edges), OrderFenwickCache(edges, impl="triton")
    if prefill:
        torch_cache.prefill(g[:, :prefill], V[:, :prefill])
        triton_cache.prefill(g[:, :prefill], V[:, :prefill])
    for t in range(prefill, N):
        a = torch_cache.step(f[:, t], g[:, t], b[:, t], V[:, t])
        c = triton_cache.step(f[:, t], g[:, t], b[:, t], V[:, t])
        torch.testing.assert_close(c, a, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(c, ref[:, t].float(), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(triton_cache.query(f[:, t], b[:, t]), c, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(triton_cache.tree1, torch_cache.tree1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(triton_cache.tree2, torch_cache.tree2, rtol=1e-5, atol=1e-5)
