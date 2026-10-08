"""Tests of causal order attention (learning/order_attention.py): prefill forward/backward, unmasked forward, and the
logarithmic-method decoding cache, all against the dense softmax(scores) @ V reference.

Run from the repo root: python -m pytest tests/test_order_attention.py -q
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.order_attention import (  # noqa: E402
    OrderCache, chunk_scan, chunk_scan_suffix, dense_order_attention, mixture_attention, order_attention,
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
    cache = OrderCache(chunk=8)
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
