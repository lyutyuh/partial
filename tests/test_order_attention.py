"""Tests of causal order attention (learning/order_attention.py): prefill forward/backward, unmasked forward, and the
decoding caches (``OrderCache``: static block + logarithmic tail in one slot buffer; ``OrderCacheSimple``: block list),
all against the dense softmax(scores) @ V reference.

Run from the repo root: python -m pytest tests/test_order_attention.py -q
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.order_attention import (  # noqa: E402
    OrderCache, OrderCacheSimple, _scan_tree, _sortkey, chunk_scan, chunk_scan_suffix, dense_order_attention,
    mixture_attention, order_attention,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _inputs(seed, B, H, N, d, scale=2.0):
    gen = torch.Generator().manual_seed(seed)
    f = (torch.randn(B, H, N, 2, generator=gen) * scale).to(DEVICE)
    g = (torch.randn(B, H, N, 2, generator=gen) * scale).to(DEVICE)
    b = (torch.randn(B, H, N, generator=gen) * scale).to(DEVICE)
    V = torch.randn(B, H, N, d, generator=gen).to(DEVICE)
    return f, g, b, V


@pytest.mark.parametrize("L,chunk", [(1, 64), (5, 2), (64, 64), (100, 16), (130, 64)])
def test_chunk_scan_matches_naive(L, chunk):
    gen = torch.Generator().manual_seed(L)
    logw = (torch.randn(3, L, generator=gen) * 3).to(DEVICE)
    logw[0, : L // 2] = float("-inf")  # excluded keys, including a fully excluded chunk
    V = torch.randn(3, L, 4, generator=gen).to(DEVICE)
    m, S = chunk_scan(logw, V, chunk)
    for i in range(L):
        w = torch.where(torch.isfinite(m[:, i]), torch.exp(logw[:, : i + 1] - m[:, i]), 0.0)
        torch.testing.assert_close(S[:, i], (w.unsqueeze(-1) * V[:, : i + 1]).sum(1), rtol=1e-5, atol=1e-5)
    m2, S2 = chunk_scan_suffix(logw, V, chunk)
    for i in range(L):
        w = torch.where(torch.isfinite(m2[:, i]), torch.exp(logw[:, i:] - m2[:, i]), 0.0)
        torch.testing.assert_close(S2[:, i], (w.unsqueeze(-1) * V[:, i:]).sum(1), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("B,H,N,d,chunk", [(1, 1, 1, 3, 64), (2, 2, 2, 3, 64), (2, 3, 7, 5, 2), (1, 2, 64, 8, 16),
                                           (2, 2, 100, 6, 64), (1, 4, 257, 16, 32)])
def test_order_attention_matches_dense(B, H, N, d, chunk, causal):
    f, g, b, V = _inputs(N + d, B, H, N, d)
    out = order_attention(f, g, b, V, causal, chunk)
    torch.testing.assert_close(out, dense_order_attention(f, g, b, V, causal), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("B,H,N,d", [(2, 2, 9, 4), (1, 3, 40, 8)])
def test_gradients_match_dense(B, H, N, d, causal):
    f, g, b, V = _inputs(11 + N, B, H, N, d)
    gout = torch.randn(B, H, N, d, generator=torch.Generator().manual_seed(1)).to(DEVICE)
    grads = []
    for fn in (lambda *a: order_attention(*a, causal, 8), lambda *a: dense_order_attention(*a, causal)):
        ts = [t.clone().requires_grad_() for t in (f, g, b, V)]
        (fn(*ts) * gout).sum().backward()
        grads.append([t.grad for t in ts])
    for a, r in zip(*grads):
        torch.testing.assert_close(a, r, rtol=1e-4, atol=1e-5)


def test_ties_and_large_magnitudes():
    f, g, b, V = _inputs(5, 2, 2, 33, 4, scale=40.0)
    f, g = f.round(), g.round()  # many exact rank ties
    torch.testing.assert_close(order_attention(f, g, b, V, True, 8), dense_order_attention(f, g, b, V), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("H,N,d,prefill", [(2, 1, 4, 0), (2, 9, 4, 0), (3, 40, 8, 0), (3, 40, 8, 13), (2, 70, 5, 64),
                                           (2, 70, 5, 70)])
def test_cache_decoding_matches_dense(H, N, d, prefill):
    """Prefill the first `prefill` tokens, then decode the rest token by token; every output must equal dense causal."""
    f, g, b, V = _inputs(N * 7 + prefill, 1, H, N, d)
    ref = dense_order_attention(f, g, b, V, True)[0]
    cache = OrderCacheSimple(chunk=8)
    if prefill:
        cache.prefill(g[0, :, :prefill], V[0, :, :prefill])
        out = order_attention(f[:, :, :prefill], g[:, :, :prefill], b[:, :, :prefill], V[:, :, :prefill], True, 8)[0]
        torch.testing.assert_close(out, ref[:, :prefill], rtol=1e-4, atol=1e-5)
    for t in range(prefill, N):
        out = cache.step(f[0, :, t], g[0, :, t], b[0, :, t], V[0, :, t])
        torch.testing.assert_close(out, ref[:, t], rtol=1e-4, atol=1e-5)
    sizes = [blk["rank"].shape[-1] for blk in cache.blocks]
    assert sum(sizes) == N - 1 and len(set(sizes)) == len(sizes)  # logarithmic method: distinct power-of-two sizes


def test_mixture_matches_dense_mixture():
    B, H, N, d, M = 1, 2, 20, 4, 3
    comps = [_inputs(s, B, H, N, d)[:3] for s in range(M)]
    V = _inputs(9, B, H, N, d)[3]
    gates = torch.randn(B, H, N, M, generator=torch.Generator().manual_seed(2)).to(DEVICE)
    w = torch.softmax(gates, -1)
    ref = sum(w[..., i:i + 1] * dense_order_attention(*comps[i], V) for i in range(M))
    torch.testing.assert_close(mixture_attention(gates, comps, V, True, 8), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("L,chunk", [(1, 64), (7, 2), (64, 8), (100, 8), (257, 16), (300, 64)])
def test_scan_tree_matches_chunk_scan(L, chunk):
    gen = torch.Generator().manual_seed(L + chunk)
    logw = (torch.randn(2, 3, L, generator=gen) * 50).to(DEVICE)  # spans of hundreds of nats inside a chunk
    logw[0, 0, : L // 2] = float("-inf")
    V = torch.randn(2, 3, L, 5, generator=gen).to(DEVICE)
    for got, ref in zip(_scan_tree(logw, V, chunk), chunk_scan(logw, V, chunk)):
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)


def _decode_inputs(seed, H, N, d, kind):
    f, g, b, V = _inputs(seed, 1, H, N, d, scale=40.0 if kind != "normal" else 2.0)
    if kind == "ties":  # integer realizers: many exact rank ties, exact logits, spans of hundreds of nats
        f, g, b = f.round(), g.round(), b.round()
    return f[0], g[0], b[0], V[0]


def _check_layout(cache):
    """Keys sorted along the whole buffer; occupancy is the binary counter; free slots hold identity states."""
    assert bool((cache.comp[:, 1:] >= cache.comp[:, :-1]).all())
    segs = cache.segments()
    assert sum(n for _, n in segs) == cache.n - 1 and segs[-1][1] == cache.ns
    assert [n for _, n in segs[:-1]] == [(1 << level) * ((cache.count >> level) & 1) for level in range(cache.L)]
    free = (cache.comp & 0xFFFFFFFF) == 0xFFFFFFFF
    assert int((~free).sum()) == cache.comp.shape[0] * (cache.n - 1 + cache.L + 2)  # keys + guards
    assert bool((cache.P2[..., -1][free] == float("-inf")).all())


@pytest.mark.parametrize("kind", ["normal", "ties", "large"])
@pytest.mark.parametrize("tail", [5, None])
@pytest.mark.parametrize("prefill", [0, 1, 2, 3, 63, 64, 65, 127, 128, 200, 257])
def test_order_cache_matches_dense(prefill, tail, kind):
    """Prefill, then decode 40 tokens (70 from scratch): every output equals dense causal and the old cache."""
    H, d = 3, 5
    N = prefill + (40 if prefill else 70)
    f, g, b, V = _decode_inputs(prefill * 3 + len(kind), H, N, d, kind)
    ref = dense_order_attention(f[None], g[None], b[None], V[None], True)[0]
    cache = OrderCache(chunk=8, **({"tail_capacity": tail} if tail else {}))
    simple = OrderCacheSimple(chunk=8)
    if prefill:
        cache.prefill(g[:, :prefill], V[:, :prefill])
        simple.prefill(g[:, :prefill], V[:, :prefill])
        _check_layout(cache)
    for t in range(prefill, N):
        out = cache.step(f[:, t], g[:, t], b[:, t], V[:, t])
        torch.testing.assert_close(out, ref[:, t], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out, simple.step(f[:, t], g[:, t], b[:, t], V[:, t]), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(cache.query(f[:, t], b[:, t]), out, rtol=0, atol=0)
        _check_layout(cache)
    if tail:
        assert cache.folds >= 2  # the tail was folded into the static block at least twice
    else:
        assert cache.folds == 0 and cache.ns == max(prefill - 1, 0)


@pytest.mark.parametrize("k", [1, 6, 8, 12])
def test_no_cascade_after_power_of_two_prefill(k):
    """A prefill of 2^k tokens is one static block: decode step i rebuilds 2^j keys (j = lowest zero bit of i - 1)."""
    H, d, N = 2, 4, (1 << k) + 8
    f, g, b, V = _decode_inputs(k, H, N, d, "normal")
    cache = OrderCache(chunk=64)
    cache.prefill(g[:, :1 << k], V[:, :1 << k])
    assert cache.rebuilt == (1 << k) - 1 and cache.segments()[-1] == (cache.L, (1 << k) - 1)
    work = []
    for t in range(1 << k, N):
        before = cache.rebuilt
        cache.step(f[:, t], g[:, t], b[:, t], V[:, t])
        work.append(cache.rebuilt - before)
    assert work == [1, 2, 1, 4, 1, 2, 1, 8]
    simple = OrderCacheSimple(chunk=64)  # the block list cascades: its first step merges all 2^k keys into one block
    simple.prefill(g[:, :1 << k], V[:, :1 << k])
    simple.step(f[:, 1 << k], g[:, 1 << k], b[:, 1 << k], V[:, 1 << k])
    assert [blk["rank"].shape[-1] for blk in simple.blocks] == [1 << k]


def test_order_cache_amortised_work_and_fold():
    """From scratch the merge work is O(T log T); a full tail is folded in one rebuild of the whole cache."""
    H, d, T = 2, 3, 300
    f, g, b, V = _decode_inputs(3, H, T, d, "normal")
    cache = OrderCache(chunk=8, tail_capacity=31, min_tail_capacity=31)
    for t in range(T):
        before, folds = cache.rebuilt, cache.folds
        cache.step(f[:, t], g[:, t], b[:, t], V[:, t])
        if cache.folds > folds:
            assert cache.rebuilt - before == t and cache.count == 0  # the whole past (sink excluded) in one block
    assert cache.folds == (T - 2) // 32 and cache.rebuilt <= T * T.bit_length() + cache.folds * T


def test_sortkey_order():
    x = torch.tensor([float("-inf"), -1e30, -2.5, -1e-40, -0.0, 0.0, 1e-40, 3.0, 1e30, float("inf")])
    k = _sortkey(x)
    assert bool((k[1:] >= k[:-1]).all()) and k[4] == k[5] and int(k.max()) < 0xFFFFFFFF


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("prefill,tail,max_level", [(0, 6, 8), (128, 1023, 2), (257, 13, 8)])
def test_order_cache_cuda_graph_matches_eager(prefill, tail, max_level):
    """Graph replays (with eager folds and, for max_level 2, eager deep merges in between) match the eager cache."""
    H, d, N = 4, 16, prefill + 300
    f, g, b, V = _decode_inputs(prefill + 1, H, N, d, "normal")
    ref = dense_order_attention(f[None], g[None], b[None], V[None], True)[0]
    eager = OrderCache(chunk=8, tail_capacity=tail)
    graph = OrderCache(chunk=8, tail_capacity=tail, cuda_graph=True, graph_max_level=max_level)
    if prefill:
        eager.prefill(g[:, :prefill], V[:, :prefill])
        graph.prefill(g[:, :prefill], V[:, :prefill])
    for t in range(prefill, N):
        out = graph.step(f[:, t], g[:, t], b[:, t], V[:, t])
        torch.testing.assert_close(out, eager.step(f[:, t], g[:, t], b[:, t], V[:, t]), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out, ref[:, t], rtol=1e-5, atol=1e-5)
    assert graph.captures > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_order_cache_graphs_share_one_pool():
    """The query graph and all step graphs of one buffer share one memory pool; a fold starts a fresh pool."""
    H, d, N = 2, 8, 100
    f, g, b, V = _decode_inputs(7, H, N, d, "normal")
    ref = dense_order_attention(f[None], g[None], b[None], V[None], True)[0]
    cache = OrderCache(chunk=8, tail_capacity=63, min_tail_capacity=63, cuda_graph=True)
    pools = {}  # fold count -> pool ids of the graphs alive after each step
    for t in range(N):
        out = cache.step(f[:, t], g[:, t], b[:, t], V[:, t])
        torch.testing.assert_close(cache.query(f[:, t], b[:, t]), out, rtol=0, atol=0)
        torch.testing.assert_close(out, ref[:, t], rtol=1e-5, atol=1e-5)
        pools.setdefault(cache.folds, set()).update(graph.pool() for graph, _ in cache._graphs.values())
    assert cache.folds == 1 and len(cache._graphs) == 1 + 6  # query graph + step graphs of levels 0..5
    assert len(pools[0]) == 1 and len(pools[1]) == 1 and pools[0] != pools[1]
