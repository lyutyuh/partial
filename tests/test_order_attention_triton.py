"""Tests of the Triton causal order attention (learning/order_attention_triton.py) against the dense softmax(scores) @ V
reference and the torch position-tree ``order_attention``.

Forward: outputs, the tensors saved for the backward (lse, A1, U1, p0), ties, large magnitudes, one-branch inputs, bf16
values, the mixture, error accumulation in long level blocks (N = 2^20 against fp64) and CUDA-graph capture.
Backward: the gradients of f, g, b and V (and of the mixture gates) against autograd of the dense reference in fp64 with
the kernels' rank rule for the branch (the tie-free truth) and of ``dense_order_attention`` in fp32 (on the
branch-invariant sums), on the same shape grid plus ties, large magnitudes, low-precision values, deep trees, N = 2^20
query blocks, fp32 central differences and CUDA-graph capture of forward + backward.
Mixture: the fused function against autograd over the per-component outputs (shape grid, gate weights that are exactly
0, input dtypes, partial requires_grad), the no-grad forward's output bits and peak memory, and CUDA-graph capture.

Run from the repo root on a GPU node: python -m pytest tests/test_order_attention_triton.py -q -p no:cacheprovider
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.order_attention import dense_order_attention, mixture_attention, order_attention  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("the Triton order attention tests need a GPU", allow_module_level=True)

from learning.order_attention_triton import (  # noqa: E402
    _order_attention_bwd, _order_attention_fwd, mixture_attention_triton, order_attention_triton,
)

DEVICE = "cuda"
TOL = dict(rtol=1e-5, atol=1e-5)


def _inputs(seed, B, H, N, d, scale=2.0, vdtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    f = (torch.randn(B, H, N, 2, generator=gen) * scale).to(DEVICE)
    g = (torch.randn(B, H, N, 2, generator=gen) * scale).to(DEVICE)
    b = (torch.randn(B, H, N, generator=gen) * scale).to(DEVICE)
    V = torch.randn(B, H, N, d, generator=gen).to(DEVICE, vdtype)
    return f, g, b, V


def _dense_stats(f, g, b, V, dtype=torch.float32, rows=None):
    """(out, lse, A1, U1, p0) of the query positions ``rows`` (default: all) from the explicit probabilities in
    ``dtype``; branch 1 by the kernels' fp32 rank rule r(y) <= a(x)."""
    N = f.shape[2]
    if rows is None:
        rows = torch.arange(N, device=f.device)
    rank = g[..., 0].float() - g[..., 1].float()
    a = (f[..., 0].float() - f[..., 1].float())[:, :, rows]
    f, g, b, V = f[:, :, rows].to(dtype), g.to(dtype), b[:, :, rows].to(dtype), V.to(dtype)
    s = (g.unsqueeze(2) - f.unsqueeze(3)).amin(dim=-1)  # (B, H, x, y)
    s = torch.cat([b.unsqueeze(-1), s[..., 1:]], dim=-1)
    pos = torch.arange(N, device=s.device)
    s = s.masked_fill(pos[None, :] > rows[:, None], float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse.unsqueeze(-1))
    br1 = rank.unsqueeze(-2) <= a.unsqueeze(-1)
    br1[..., 0] = False  # the sink is on no branch
    p1 = torch.where(br1, p, 0.0)
    return p @ V, lse, p1.sum(-1), p1 @ V, p[..., 0]


def _check_stats(f, g, b, V, chunk, tol=TOL):
    got = _order_attention_fwd(f, g, b, V, chunk)
    for name, x, ref in zip(("out", "lse", "A1", "U1", "p0"), got, _dense_stats(f, g, b, V)):
        torch.testing.assert_close(x, ref.float(), **tol, msg=lambda m: f"{name}: {m}")
    lse, lerr = got[1], got[5]
    assert torch.equal(lse + lerr, lse) and torch.isfinite(lerr).all()  # the residual is below half an ulp of lse
    return got[:5]


@pytest.mark.parametrize("N", [1, 2, 15, 63, 64, 65, 127, 128, 200, 1000, 4097])
@pytest.mark.parametrize("B,H", [(1, 1), (2, 3)])
@pytest.mark.parametrize("d", [16, 64, 128])
def test_matches_dense_and_torch(N, B, H, d):
    f, g, b, V = _inputs(N * 131 + d + B, B, H, N, d)
    dense = dense_order_attention(f, g, b, V)
    tree = order_attention(f, g, b, V, True, 64)
    for chunk in (16, 64):
        out = order_attention_triton(f, g, b, V, chunk)
        assert out.shape == (B, H, N, d) and out.dtype == torch.float32
        torch.testing.assert_close(out, dense, **TOL)
        torch.testing.assert_close(out, tree, **TOL)


@pytest.mark.parametrize("N,chunk", [(1, 16), (17, 16), (200, 16), (1000, 64), (4097, 64), (3000, 128)])
def test_saved_tensors_match_dense(N, chunk):
    f, g, b, V = _inputs(N + chunk, 2, 3, N, 64)
    out, lse, A1, U1, p0 = _check_stats(f, g, b, V, chunk)
    torch.testing.assert_close(out[:, :, 0], V[:, :, 0], rtol=0, atol=1e-6)  # query 0 sees only the sink
    torch.testing.assert_close(p0[:, :, 0], torch.ones_like(p0[:, :, 0]), rtol=0, atol=1e-6)
    # branch 2 from the saved tensors: A2 = 1 - A1 - p0 >= 0, U2 = out - U1 - p0 v(0)
    assert float((1 - A1 - p0).min()) > -1e-5


@pytest.mark.parametrize("N,chunk", [(65, 16), (1000, 64), (4097, 64)])
def test_ties(N, chunk):
    """Integer realizers: many exact rank ties between keys and between keys and queries, all logits exact."""
    f, g, b, V = _inputs(7 + N, 2, 3, N, 64, scale=4.0)
    f, g, b = f.round(), g.round(), b.round()
    _check_stats(f, g, b, V, chunk)
    torch.testing.assert_close(order_attention_triton(f, g, b, V, chunk), order_attention(f, g, b, V, True, 64), **TOL)


@pytest.mark.parametrize("rounded", [False, True])
@pytest.mark.parametrize("N,chunk", [(200, 16), (2000, 64), (4097, 64)])
def test_large_magnitudes(N, chunk, rounded):
    """Scale 50: logits spanning hundreds of nats inside one chunk and one tree block (fp32 underflows at ~87).

    Integer inputs make every logit exact: 1e-5 against the fp32 dense reference and the fp64 truth. With non-integer
    inputs a logit of magnitude ~300 itself rounds by ~2e-5 in fp32, which moves the fp32 dense reference (and the torch
    tree) as far from the fp64 truth (1.3e-5) as these kernels, so they are checked against the fp64 truth at 2e-5.
    """
    f, g, b, V = _inputs(11 + N, 2, 3, N, 64, scale=50.0)
    if rounded:
        f, g, b = f.round(), g.round(), b.round()
    span = (g[..., 0] - g[..., 0].amin(-1, keepdim=True)).amax()
    assert float(span) > 300
    got = _check_stats(f, g, b, V, chunk) if rounded else _order_attention_fwd(f, g, b, V, chunk)
    tol = TOL if rounded else dict(rtol=2e-5, atol=2e-5)
    for name, x, ref in zip(("out", "lse", "A1", "U1", "p0"), got, _dense_stats(f, g, b, V, torch.float64)):
        torch.testing.assert_close(x, ref.float(), **tol, msg=lambda m: f"{name}: {m}")
    assert torch.isfinite(got[0]).all()


@pytest.mark.parametrize("branch", [1, 2])
@pytest.mark.parametrize("N,chunk", [(130, 16), (2000, 64)])
def test_one_branch(N, chunk, branch):
    """All keys on branch 1 (r(y) far below every a(x)) or all on branch 2 (far above): one side of every split."""
    f, g, b, V = _inputs(3 + N + branch, 2, 3, N, 32)
    g = g.clone()
    g[..., 2 - branch] += 100.0  # branch 1: raise g2 (r -> -100); branch 2: raise g1 (r -> +100)
    out, lse, A1, U1, p0 = _check_stats(f, g, b, V, chunk)
    if branch == 1:
        torch.testing.assert_close(A1, 1 - p0, **TOL)
        torch.testing.assert_close(U1, out - p0.unsqueeze(-1) * V[:, :, :1], **TOL)
    else:
        assert float(A1.abs().max()) == 0.0 and float(U1.abs().max()) == 0.0


@pytest.mark.parametrize("vdtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("N,chunk", [(100, 16), (4097, 64)])
def test_low_precision_values(N, chunk, vdtype):
    """bf16 / fp16 values are read as they are and accumulated in fp32: same output as the fp32 reference of V."""
    f, g, b, V = _inputs(5 + N, 2, 3, N, 128, vdtype=vdtype)
    out = order_attention_triton(f, g, b, V, chunk)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, dense_order_attention(f, g, b, V), **TOL)
    _check_stats(f, g, b, V, chunk)


@pytest.mark.parametrize("d", [8, 80])
def test_odd_value_dims(d):
    f, g, b, V = _inputs(d, 1, 2, 300, d)
    torch.testing.assert_close(order_attention_triton(f, g, b, V, 16), dense_order_attention(f, g, b, V), **TOL)


def test_noncontiguous_inputs():
    f, g, b, V = _inputs(9, 2, 3, 150, 64)
    ft, gt = f.transpose(1, 2).contiguous().transpose(1, 2), g.transpose(1, 2).contiguous().transpose(1, 2)
    Vt = V.transpose(1, 2).contiguous().transpose(1, 2)
    assert not ft.is_contiguous() and not Vt.is_contiguous()
    torch.testing.assert_close(order_attention_triton(ft, gt, b, Vt, 16), dense_order_attention(f, g, b, V), **TOL)


@pytest.mark.parametrize("N,chunk", [(20, 16), (700, 64)])
def test_mixture_matches_reference(N, chunk):
    B, H, d, M = 2, 2, 32, 3
    comps = [_inputs(s, B, H, N, d)[:3] for s in range(M)]
    V = _inputs(9, B, H, N, d)[3]
    gates = torch.randn(B, H, N, M, generator=torch.Generator().manual_seed(2)).to(DEVICE)
    w = torch.softmax(gates, -1)
    dense = sum(w[..., i:i + 1] * dense_order_attention(*comps[i], V) for i in range(M))
    out = mixture_attention_triton(gates, comps, V, chunk)
    torch.testing.assert_close(out, dense, **TOL)
    torch.testing.assert_close(out, mixture_attention(gates, comps, V, True, 64), **TOL)


def _dense_rows(f, g, b, V, rows):
    """Dense causal output of the query positions ``rows`` only: O(len(rows) N) memory for long sequences."""
    f, g, b, V = f.float(), g.float(), b.float(), V.float()
    fx = f[:, :, rows]  # (B, H, R, 2)
    s = (g.unsqueeze(2) - fx.unsqueeze(3)).amin(dim=-1)  # (B, H, R, N)
    s = torch.cat([b[:, :, rows].unsqueeze(-1), s[..., 1:]], dim=-1)
    pos = torch.arange(s.shape[-1], device=s.device)
    s = s.masked_fill(pos[None, None, None, :] > rows[None, None, :, None], float("-inf"))
    return torch.softmax(s, dim=-1) @ V


@pytest.mark.parametrize("N,chunk", [(65536, 64), (40000, 16)])
def test_long_sequence_rows(N, chunk):
    """Deep trees (10 / 12 levels, blocks up to 32768 keys) vs dense rows: queries at and before every 37th chunk
    boundary, 200 random ones and the last."""
    f, g, b, V = _inputs(N, 1, 2, N, 64)
    out = order_attention_triton(f, g, b, V, chunk)
    gen = torch.Generator().manual_seed(0)
    bounds = torch.arange(0, N, chunk * 37)
    rows = torch.cat([bounds, (bounds - 1).clamp(min=0), torch.randint(0, N, (200,), generator=gen),
                      torch.tensor([N - 1])]).unique().to(DEVICE)
    for part in rows.split(256):
        torch.testing.assert_close(out[:, :, part], _dense_rows(f, g, b, V, part), **TOL)


def _long_block_inputs(case, N, H=1, d=32):
    """Keys whose g grows along the rank order, so a state's running max moves at nearly every slot of a level block.

    ``const_g1`` / ``const_g2``: one key realizer constant (dead or bias-only), so the rank is the other one up to a
    constant. ``slope_g1`` / ``slope_g2``: g = +-1e-9 * position on one realizer and 0 on the other (an ALiBi-like
    recency score): near-equal terms whose rescale factors are all the same, with v correlated with the position.
    """
    gen = torch.Generator().manual_seed(len(case))
    y = torch.arange(N, dtype=torch.float32)
    f = torch.randn(1, H, N, 2, generator=gen) * 0.1
    g = torch.randn(1, H, N, 2, generator=gen) * 0.1
    b = torch.randn(1, H, N, generator=gen) * 0.1
    V = torch.randn(1, H, N, d, generator=gen)
    if case.startswith("const"):
        g[..., int(case[-1]) - 1] = 0.25
    elif case == "slope_g1":
        g[..., 0], g[..., 1], V[..., 0] = 1e-9 * y, 0.0, y / N
    else:
        g[..., 0], g[..., 1], V[..., 0] = 0.0, -1e-9 * y, -y / N
    return [t.to(DEVICE) for t in (f, g, b, V)]


@pytest.mark.parametrize("case", ["const_g1", "const_g2", "slope_g1", "slope_g2"])
def test_long_block_accumulation(case):
    """N = 2^20 (14 levels, top blocks of 2^19 keys) vs fp64 dense rows at 1e-5: the scan's rescale roundings and carry
    additions must not accumulate with the block length (with per-record rescaling: lse / A1 at 9x / 11x the tolerance
    for a constant realizer; with a direct carry factor but no whole-nat reference and no compensated carry: 2-3x for
    the slopes)."""
    N = 1 << 20
    f, g, b, V = _long_block_inputs(case, N)
    got = _order_attention_fwd(f, g, b, V, 64)
    gen = torch.Generator().manual_seed(1)
    rows = torch.cat([torch.arange(N - 32, N), torch.randint(N // 2, N, (32,), generator=gen)]).unique().to(DEVICE)
    for name, x, ref in zip(("out", "lse", "A1", "U1", "p0"), got, _dense_stats(f, g, b, V, torch.float64, rows)):
        torch.testing.assert_close(x[:, :, rows], ref.float(), **TOL, msg=lambda m: f"{name}: {m}")


def test_cuda_graph_capture():
    """After one call with the same shapes the forward needs no host sync or copy: it replays from a CUDA graph."""
    static = [t.clone() for t in _inputs(1, 2, 3, 1000, 64)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        order_attention_triton(*static, 64)  # compile and warm up outside the capture
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = order_attention_triton(*static, 64)
    for seed in (2, 3):
        fresh = _inputs(seed, 2, 3, 1000, 64)
        for dst, src in zip(static, fresh):
            dst.copy_(src)
        graph.replay()
        torch.testing.assert_close(out, dense_order_attention(*fresh), **TOL)


def test_cuda_graph_first_call_and_churn():
    """A shape first seen inside the capture (kernels compiled by another shape) captures, since the forward makes no
    host-to-device copy, and the graph stays valid after 299 other shapes and fresh allocations, since it reads no
    cached device tensor that could be freed and reused (an lru-cached offset table did both)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        order_attention_triton(*_inputs(1, 2, 3, 1000, 64), 64)  # compile the kernels on another shape
    torch.cuda.current_stream().wait_stream(s)
    static = [t.clone() for t in _inputs(2, 1, 5, 1000, 64)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = order_attention_triton(*static, 64)
    for bh in range(1, 300):
        _order_attention_fwd(*_inputs(bh, bh, 1, 200, 64), 64)
    torch.cuda.synchronize()
    junk = [torch.zeros(64, dtype=torch.int64, device=DEVICE) for _ in range(4000)]  # reuse freed blocks
    for seed in (3, 4):
        fresh = _inputs(seed, 1, 5, 1000, 64)
        for dst, src in zip(static, fresh):
            dst.copy_(src)
        graph.replay()
        torch.testing.assert_close(out, dense_order_attention(*fresh), **TOL)
    del junk


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_non_current_device():
    f, g, b, V = (t.to("cuda:1") for t in _inputs(4, 1, 2, 300, 32))
    assert torch.cuda.current_device() == 0
    torch.testing.assert_close(order_attention_triton(f, g, b, V, 16), dense_order_attention(f, g, b, V), **TOL)
    gates = torch.zeros(1, 2, 300, 2, device="cuda:1")
    torch.testing.assert_close(mixture_attention_triton(gates, [(f, g, b)] * 2, V, 16),
                               dense_order_attention(f, g, b, V), **TOL)


def test_double_backward_raises():
    """The backward is a Triton kernel chain, not differentiable itself: create_graph must fail loudly."""
    f, g, b, V = (t.requires_grad_() for t in _inputs(1, 1, 2, 40, 16))
    out = order_attention_triton(f, g, b, V, 16)
    (gf,) = torch.autograd.grad(out.sum(), f, create_graph=True)
    with pytest.raises(RuntimeError):
        gf.sum().backward()


def test_input_validation():
    f, g, b, V = _inputs(1, 1, 1, 10, 16)
    with pytest.raises(ValueError):
        order_attention_triton(f, g, b, V, 24)  # not a power of two
    with pytest.raises(ValueError):
        order_attention_triton(f, g, b, V, 8)  # below 16
    with pytest.raises(ValueError):
        order_attention_triton(f[..., :1], g, b, V, 16)


# ----------------------------------------------------------------------------------------------------------------------
# Backward
# ----------------------------------------------------------------------------------------------------------------------

# Gradients are compared at an absolute tolerance relative to the reference gradient's scale: max |error| <=
# GRAD_TOL max(1, max |reference|). Autograd of the fp32 dense reference itself reaches about 3e-6 of the scale against
# fp64 (dV at N = 4097: its per-key sums run over thousands of queries).
GRAD_TOL = 5e-6


def _upstream(seed, shape):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed + 1000)).to(DEVICE)


def _rank_dense(f, g, b, V, dtype=torch.float64):
    """Dense causal order attention in ``dtype`` with every pair's branch picked by the kernels' fp32 rank rule
    r(y) <= a(x): smooth along that choice, so its fp64 autograd is the tie-free reference."""
    N = f.shape[2]
    rank = (g[..., 0].float() - g[..., 1].float()).detach()
    a = (f[..., 0].float() - f[..., 1].float()).detach()
    br1 = rank.unsqueeze(-2) <= a.unsqueeze(-1)  # (B, H, x, y)
    f, g, b, V = f.to(dtype), g.to(dtype), b.to(dtype), V.to(dtype)
    s = torch.where(br1, g[..., 0].unsqueeze(-2) - f[..., 0].unsqueeze(-1),
                    g[..., 1].unsqueeze(-2) - f[..., 1].unsqueeze(-1))
    s = torch.cat([b.unsqueeze(-1), s[..., 1:]], dim=-1)
    s = s.masked_fill(~torch.ones(N, N, dtype=torch.bool, device=s.device).tril(), float("-inf"))
    return torch.softmax(s, dim=-1) @ V


def _grads(fn, inputs, dO):
    """Gradients of <dO, fn(*inputs)> with respect to every input (fresh leaf copies)."""
    xs = [t.detach().clone().requires_grad_() for t in inputs]
    out = fn(*xs)
    out.backward(dO.to(out.dtype))
    return [x.grad for x in xs]


def _assert_grads(got, ref, tol=GRAD_TOL, names=("df", "dg", "db", "dV")):
    for name, x, r in zip(names, got, ref):
        assert x.shape == r.shape and bool(torch.isfinite(x).all()), name
        scale = max(1.0, float(r.abs().max()))
        torch.testing.assert_close(x.double(), r.double(), rtol=0, atol=tol * scale, msg=lambda m: f"{name}: {m}")


def _fold(gr):
    """The branch-invariant gradients: df1 + df2, dg1 + dg2, db, dV."""
    return [gr[0].sum(-1), gr[1].sum(-1), gr[2], gr[3]]


def _check_grads(f, g, b, V, chunks, dO, tol=GRAD_TOL, dense32=True):
    """Triton gradients (for each chunk size) vs autograd of the fp64 rank-rule reference (all of them) and of the
    fp32 ``dense_order_attention`` (the branch-invariant sums only: torch.amin shares a tied gradient between the two
    branches, the kernels give it to branch 1). Returns the gradients of the last chunk size."""
    ref = _grads(_rank_dense, (f, g, b, V), dO)
    ref32 = _fold(_grads(dense_order_attention, (f, g, b, V), dO)) if dense32 else None
    for chunk in chunks:
        got = _grads(lambda *a: order_attention_triton(*a, chunk), (f, g, b, V), dO)
        _assert_grads(got, ref, tol)
        if dense32:
            _assert_grads(_fold(got), ref32, 2 * tol, ("df1+df2", "dg1+dg2", "db", "dV"))
    return got


@pytest.mark.parametrize("N", [1, 2, 15, 63, 64, 65, 127, 128, 200, 1000, 4097])
@pytest.mark.parametrize("B,H", [(1, 1), (2, 3)])
@pytest.mark.parametrize("d", [16, 64, 128])
def test_grads_match_dense(N, B, H, d):
    f, g, b, V = _inputs(N * 131 + d + B, B, H, N, d)
    dO = _upstream(N + d + B, (B, H, N, d))
    df, dg, db, dV = _check_grads(f, g, b, V, (16, 64), dO)
    assert float(dg[:, :, 0].abs().max()) == 0.0  # key 0 never takes an order logit


@pytest.mark.parametrize("N,chunk", [(17, 16), (1000, 32), (3000, 128)])
def test_grads_other_chunks(N, chunk):
    f, g, b, V = _inputs(N + chunk, 2, 3, N, 64)
    _check_grads(f, g, b, V, (chunk,), _upstream(N, (2, 3, N, 64)))


@pytest.mark.parametrize("N,chunk", [(65, 16), (1000, 64), (4097, 64)])
def test_grad_ties(N, chunk):
    """Integer realizers: many exact rank ties between keys and between keys and queries."""
    f, g, b, V = _inputs(7 + N, 2, 3, N, 64, scale=4.0)
    f, g, b = f.round(), g.round(), b.round()
    _check_grads(f, g, b, V, (chunk,), _upstream(7 + N, (2, 3, N, 64)))


@pytest.mark.parametrize("rounded", [False, True])
@pytest.mark.parametrize("N,chunk", [(200, 16), (2000, 64), (4097, 64)])
def test_grad_large_magnitudes(N, chunk, rounded):
    """Scale 50: logits spanning hundreds of nats. Integer inputs (exact logits) at the usual tolerance. With
    non-integer inputs the fp32 logits themselves round by ~2e-5 at |logit| ~ 300 (and the forward's lse with them),
    so the check against the fp64 truth is at 2 GRAD_TOL."""
    f, g, b, V = _inputs(11 + N, 2, 3, N, 64, scale=50.0)
    if rounded:
        f, g, b = f.round(), g.round(), b.round()
    _check_grads(f, g, b, V, (chunk,), _upstream(11 + N, (2, 3, N, 64)), tol=GRAD_TOL if rounded else 2 * GRAD_TOL)


@pytest.mark.parametrize("branch", [1, 2])
@pytest.mark.parametrize("N,chunk", [(130, 16), (2000, 64)])
def test_grad_one_branch(N, chunk, branch):
    """All keys on branch 1 or all on branch 2: the other branch's df / dg vanish (exactly where nothing is summed)."""
    f, g, b, V = _inputs(3 + N + branch, 2, 3, N, 32)
    g = g.clone()
    g[..., 2 - branch] += 100.0
    df, dg, db, dV = _check_grads(f, g, b, V, (chunk,), _upstream(N, (2, 3, N, 32)))
    k = 2 - branch  # the unused branch
    assert float(dg[..., k].abs().max()) == 0.0
    if branch == 2:
        assert float(df[..., 0].abs().max()) == 0.0  # A1 = U1 = 0 exactly
    else:
        torch.testing.assert_close(df[..., 1], torch.zeros_like(df[..., 1]), rtol=0, atol=1e-5)


@pytest.mark.parametrize("vdtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("N,chunk", [(100, 16), (4097, 64)])
def test_grad_low_precision_values(N, chunk, vdtype):
    """bf16 / fp16 values: df, dg, db and the fp32 dV of the kernels at the usual tolerance (against the fp64 reference
    of the same values); the returned dV is that, rounded to the value dtype."""
    f, g, b, V = _inputs(5 + N, 2, 3, N, 128, vdtype=vdtype)
    dO = _upstream(5 + N, (2, 3, N, 128))
    got = _grads(lambda *a: order_attention_triton(*a, chunk), (f, g, b, V), dO)
    ref = _grads(_rank_dense, (f, g, b, V.float()), dO)
    assert got[3].dtype == vdtype
    _assert_grads(got[:3], ref[:3])
    dV32 = _order_attention_bwd(f, g, b, V, *_order_attention_fwd(f, g, b, V, chunk), dO, chunk)[3]
    _assert_grads([dV32], ref[3:], names=("dV fp32",))
    torch.testing.assert_close(got[3], dV32.to(vdtype), rtol=0, atol=0)


@pytest.mark.parametrize("d", [8, 80, 192, 256])
def test_grad_other_value_dims(d):
    """Odd d, and d above 128, where the intra tile splits d across programs."""
    f, g, b, V = _inputs(d, 1, 2, 300, d)
    _check_grads(f, g, b, V, (16, 64), _upstream(d, (1, 2, 300, d)))


def test_grad_noncontiguous_inputs_and_upstream():
    f, g, b, V = _inputs(9, 2, 3, 150, 64)
    ft, gt = f.transpose(1, 2).contiguous().transpose(1, 2), g.transpose(1, 2).contiguous().transpose(1, 2)
    Vt = V.transpose(1, 2).contiguous().transpose(1, 2)
    dO = _upstream(9, (2, 150, 3, 64)).transpose(1, 2)
    assert not ft.is_contiguous() and not Vt.is_contiguous() and not dO.is_contiguous()
    got = _grads(lambda *a: order_attention_triton(*a, 16), (ft, gt, b, Vt), dO)
    _assert_grads(got, _grads(_rank_dense, (f, g, b, V), dO.contiguous()))


def test_grad_input_dtypes_and_partial_requires_grad():
    """bf16 realizers and values get bf16 gradients (the fp32 gradients of the same values, rounded); inputs without
    requires_grad are skipped."""
    f, g, b, V = _inputs(13, 1, 2, 200, 32)
    dO = _upstream(13, (1, 2, 200, 32))
    lo = [t.bfloat16() for t in (f, g, b, V)]
    got = _grads(lambda *a: order_attention_triton(*a, 16), lo, dO)
    ref = _grads(_rank_dense, [t.float() for t in lo], dO)
    for x, r in zip(got, ref):
        assert x.dtype == torch.bfloat16
        torch.testing.assert_close(x, r.bfloat16())
    V2 = V.clone().requires_grad_()
    order_attention_triton(f, g, b, V2, 16).backward(dO)
    torch.testing.assert_close(V2.grad.double(), _grads(_rank_dense, (f, g, b, V), dO)[3].double(), rtol=0, atol=1e-5)


def _dense_grads_rows(f, g, b, V, dO, rows_per=512):
    """fp64 gradients of <dO, out> by the textbook softmax backward, a block of queries at a time (O(rows N) memory):
    p = softmax(s), D = <dO, p V>, ds = p (dO V^T - D); df_k = -sum of ds over branch k, db = ds(x, 0),
    dg_k(y) = sum_x of ds over the queries with y on branch k, dV = p^T dO."""
    B, H, N, _ = f.shape
    rank = g[..., 0].float() - g[..., 1].float()
    a = f[..., 0].float() - f[..., 1].float()
    f, g, b, V, dO = (t.double() for t in (f, g, b, V, dO))
    df, dg = torch.zeros_like(f), torch.zeros_like(g)
    db, dV = torch.zeros_like(b), torch.zeros_like(V)
    keys = torch.arange(N, device=f.device)
    for x0 in range(0, N, rows_per):
        xs = torch.arange(x0, min(N, x0 + rows_per), device=f.device)
        br1 = rank[:, :, None, :] <= a[:, :, xs, None]  # (B, H, R, N)
        s = torch.where(br1, g[:, :, None, :, 0] - f[:, :, xs, None, 0], g[:, :, None, :, 1] - f[:, :, xs, None, 1])
        s[..., 0] = b[:, :, xs]
        p = torch.softmax(s.masked_fill(keys[None, :] > xs[:, None], float("-inf")), dim=-1)
        D = (dO[:, :, xs] * (p @ V)).sum(-1, keepdim=True)
        ds = p * (dO[:, :, xs] @ V.transpose(-1, -2) - D)
        db[:, :, xs] = ds[..., 0]
        ds[..., 0] = 0.0
        ds1 = torch.where(br1, ds, 0.0)
        ds2 = ds - ds1
        df[:, :, xs, 0], df[:, :, xs, 1] = -ds1.sum(-1), -ds2.sum(-1)
        dg[..., 0] += ds1.sum(-2)
        dg[..., 1] += ds2.sum(-2)
        dV += p.transpose(-1, -2) @ dO[:, :, xs]
    return df, dg, db, dV


@pytest.mark.parametrize("N,chunk", [(65536, 64), (40000, 16)])
def test_grad_long_sequence(N, chunk):
    """Deep transposed trees (10 / 12 levels, query blocks up to 32768 / 32768 rows; for N = 40000 the top odd block
    runs past N) against the full fp64 gradients."""
    f, g, b, V = _inputs(N, 1, 2, N, 32)
    dO = _upstream(N, (1, 2, N, 32))
    got = _grads(lambda *a: order_attention_triton(*a, chunk), (f, g, b, V), dO)
    _assert_grads(got, _dense_grads_rows(f, g, b, V, dO))


def _key_grads_given_forward(f, g, V, dO, lse, lerr, out, keys):
    """fp64 dV and dg of the keys ``keys`` (each summed over all queries x >= y), with the probabilities and D(x) taken
    from the forward's lse (+ residual) and out: a check of the backward's accumulation over long query blocks."""
    N = f.shape[2]
    a = (f[..., 0].float() - f[..., 1].float())[0, 0]
    r = (g[..., 0].float() - g[..., 1].float())[0, 0, keys]
    f, g, V, dO = (t[0, 0].double() for t in (f, g, V, dO))
    L = lse[0, 0].double() + lerr[0, 0].double()
    D = (dO * out[0, 0].double()).sum(-1, keepdim=True)  # (N, 1)
    br1 = r[None, :] <= a[:, None]  # (N, K)
    s = torch.where(br1, g[keys, 0][None, :] - f[:, 0:1], g[keys, 1][None, :] - f[:, 1:2])
    vis = torch.arange(N, device=f.device)[:, None] >= keys[None, :]
    p = torch.where(vis, torch.exp(s - L[:, None]), 0.0)
    ds = p * (dO @ V[keys].T - D)
    dg = torch.stack([torch.where(br1, ds, 0.0).sum(0), torch.where(br1, 0.0, ds).sum(0)], -1)
    return p.T @ dO, dg


@pytest.mark.parametrize("case", ["const_g1", "slope_g1", "slope_g2", "slope_f1", "slope_f2"])
def test_grad_long_block_accumulation(case):
    """N = 2^20 (14 levels, query blocks of up to 2^19 rows): dV and dg of early keys (which read the longest query
    blocks) and of random keys, against fp64 sums given the forward's lse and out. Besides the forward's key-side cases,
    ``slope_f1`` / ``slope_f2`` make the query side drift (f = +-1e-5 * position on one realizer, so a query's rank is
    its position and its branch log-weight -f - lse moves at nearly every slot of a sorted query block)."""
    N = 1 << 20
    if case.startswith("slope_f"):
        f, g, b, V = _long_block_inputs("const_g1", N)
        y = torch.arange(N, dtype=torch.float32, device=DEVICE)
        f = f.clone()
        k = int(case[-1]) - 1
        f[..., k], f[..., 1 - k] = (1e-5 if k == 0 else -1e-5) * y, 0.0
        V = V.clone()
        V[..., 0] = y / N
    else:
        f, g, b, V = _long_block_inputs(case, N)
    dO = _upstream(len(case), V.shape)
    out, lse, A1, U1, p0, lerr = _order_attention_fwd(f, g, b, V, 64)
    df, dg, db, dV = _order_attention_bwd(f, g, b, V, out, lse, A1, U1, p0, lerr, dO, 64)
    gen = torch.Generator().manual_seed(2)
    keys = torch.cat([torch.arange(1, 33), torch.randint(1, N // 2, (32,), generator=gen)]).unique().to(DEVICE)
    rdV, rdg = _key_grads_given_forward(f, g, V, dO, lse, lerr, out, keys)
    _assert_grads([dV[0, 0, keys], dg[0, 0, keys]], [rdV, rdg], names=("dV", "dg"))


@pytest.mark.parametrize("N,chunk", [(20, 16), (700, 64)])
def test_mixture_grads(N, chunk):
    """Gradients of the gates, the shared values and every component's f, g, b against the fp64 dense mixture."""
    B, H, d, M = 2, 2, 32, 3
    comps = [_inputs(s, B, H, N, d)[:3] for s in range(M)]
    V = _inputs(9, B, H, N, d)[3]
    gates = torch.randn(B, H, N, M, generator=torch.Generator().manual_seed(2)).to(DEVICE)
    dO = _upstream(N, (B, H, N, d))

    def mix(attend):
        def run(gates, V, *flat):
            cs = [flat[3 * i:3 * i + 3] for i in range(M)]
            return attend(gates, cs, V)
        return run

    dense = mix(lambda gt, cs, V: sum(torch.softmax(gt.double(), -1)[..., i:i + 1] * _rank_dense(*cs[i], V)
                                      for i in range(M)))
    inputs = [gates, V] + [t for c in comps for t in c]
    got = _grads(mix(lambda gt, cs, V: mixture_attention_triton(gt, cs, V, chunk)), inputs, dO)
    names = ["dgates", "dV"] + [f"d{n}{i}" for i in range(M) for n in "fgb"]
    _assert_grads(got, _grads(dense, inputs, dO), names=names)


def _autograd_mixture(gates, comps, V, chunk):
    """The mixture as autograd over the per-component Triton outputs (the implementation before the fused function)."""
    w = torch.softmax(gates.float(), dim=-1)
    out = 0
    for i, (f, g, b) in enumerate(comps):
        out = out + w[..., i:i + 1] * order_attention_triton(f, g, b, V, chunk)
    return out


def _mix_inputs(seed, B, H, N, d, M, gscale=1.0):
    comps = [_inputs(seed + 17 * m, B, H, N, d)[:3] for m in range(M)]
    V = _inputs(seed + 1, B, H, N, d)[3]
    gates = (torch.randn(B, H, N, M, generator=torch.Generator().manual_seed(seed + 2)) * gscale).to(DEVICE)
    return [gates, V] + [t for c in comps for t in c]


def _mix_fn(attend, chunk):
    """attend(gates, comps, V, chunk) as a function of the flat inputs (gates, V, f0, g0, b0, f1, ...)."""
    def run(gates, V, *flat):
        return attend(gates, [flat[i:i + 3] for i in range(0, len(flat), 3)], V, chunk)
    return run


def _mix_names(M):
    return ["dgates", "dV"] + [f"d{n}{i}" for i in range(M) for n in "fgb"]


@pytest.mark.parametrize("B,H,N,d,M,chunk", [
    (1, 1, 1, 16, 2, 16),  # only the sink
    (2, 3, 17, 16, 3, 16),  # two chunks
    (1, 2, 64, 32, 2, 64),  # one chunk: no tree
    (2, 3, 1000, 128, 4, 64),  # 16 chunks
    (1, 2, 4097, 128, 4, 64),  # a chunk count that is not a power of two
    (1, 2, 1500, 192, 2, 32),  # d above 128: the intra tile splits d
    (1, 2, 300, 8, 3, 16),  # d below the tl.dot minimum
    (2, 2, 700, 64, 1, 64),  # one component
])
def test_mixture_fused_matches_autograd(B, H, N, d, M, chunk):
    """The fused mixture function against autograd over the per-component outputs: the output to fp32 rounding and
    every gradient (gates, V, each component's f, g, b) at the usual tolerance; the no-grad forward, which adds and
    drops one component at a time, gives the same output bits as the grad forward."""
    inputs = _mix_inputs(N + d + M, B, H, N, d, M)
    dO = _upstream(N + M, (B, H, N, d))
    got = _grads(_mix_fn(mixture_attention_triton, chunk), inputs, dO)
    _assert_grads(got, _grads(_mix_fn(_autograd_mixture, chunk), inputs, dO), names=_mix_names(M))
    out = _mix_fn(mixture_attention_triton, chunk)(*[t.clone().requires_grad_() for t in inputs])
    assert out.requires_grad
    with torch.no_grad():
        out_ng = _mix_fn(mixture_attention_triton, chunk)(*inputs)
        torch.testing.assert_close(out_ng, _mix_fn(_autograd_mixture, chunk)(*inputs), rtol=1e-6, atol=1e-6)
    assert torch.equal(out_ng, out.detach())
    assert torch.equal(_mix_fn(mixture_attention_triton, chunk)(*inputs), out_ng)  # nothing requires grad: no-grad path


@pytest.mark.parametrize("case", ["underflow", "neg_inf"])
def test_mixture_zero_weights(case):
    """Gate weights that are exactly 0: gates at scale 200 (the softmax underflows for most (query, component) pairs),
    or one component's gates at -inf (its weight is 0 everywhere, so its gradients and its gate gradient are 0)."""
    M = 3
    inputs = _mix_inputs(31, 2, 2, 700, 32, M, gscale=200.0 if case == "underflow" else 1.0)
    if case == "neg_inf":
        inputs[0] = inputs[0].clone()
        inputs[0][..., 1] = float("-inf")
    w = torch.softmax(inputs[0], -1)
    assert float((w == 0).float().mean()) > 0.3
    dO = _upstream(31, (2, 2, 700, 32))
    got = _grads(_mix_fn(mixture_attention_triton, 64), inputs, dO)
    _assert_grads(got, _grads(_mix_fn(_autograd_mixture, 64), inputs, dO), names=_mix_names(M))
    if case == "neg_inf":
        assert float(got[0][..., 1].abs().max()) == 0.0
        for t in got[5:8]:  # f1, g1, b1
            assert float(t.abs().max()) == 0.0


def test_mixture_dtypes_and_partial_requires_grad():
    """bf16 gates, realizers and values: gradients in the input dtypes, the fp32 gradients of the same values rounded.
    Inputs without requires_grad get no gradient, and those that require it get the same bits as when all do."""
    M, chunk = 3, 16
    inputs = _mix_inputs(41, 1, 2, 200, 32, M)
    dO = _upstream(41, (1, 2, 200, 32))
    lo = [t.bfloat16() for t in inputs]
    got = _grads(_mix_fn(mixture_attention_triton, chunk), lo, dO)
    ref = _grads(_mix_fn(mixture_attention_triton, chunk), [t.float() for t in lo], dO)
    for name, x, r in zip(_mix_names(M), got, ref):
        assert x.dtype == torch.bfloat16, name
        torch.testing.assert_close(x, r.bfloat16(), msg=lambda m: f"{name}: {m}")
    full = _grads(_mix_fn(mixture_attention_triton, chunk), inputs, dO)
    for which in ([0], [1], [5, 6, 7], [2, 9]):  # gates only, V only, component 1 only, f0 and g2
        xs = [t.clone().requires_grad_(i in which) for i, t in enumerate(inputs)]
        _mix_fn(mixture_attention_triton, chunk)(*xs).backward(dO)
        for i, x in enumerate(xs):
            if i in which:
                assert torch.equal(x.grad, full[i]), (which, i)
            else:
                assert x.grad is None, (which, i)


def test_mixture_nograd_memory():
    """Without gradients the forward keeps only one component's tensors alive at a time: its peak memory is below the
    grad-mode forward's, which keeps every component's output and U1 for the backward (2 (M - 1) outputs more, plus the
    per-query statistics; checked at half that)."""
    B, H, N, d, M = 1, 4, 4096, 64, 4
    inputs = _mix_inputs(51, B, H, N, d, M)
    fn = _mix_fn(mixture_attention_triton, 64)
    leaves = [t.clone().requires_grad_() for t in inputs]
    fn(*leaves)  # compile both paths
    fn(*inputs)

    def peak(*args):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = fn(*args)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base, out

    p_nograd, _ = peak(*inputs)
    p_grad, out = peak(*leaves)
    assert out.requires_grad
    assert p_nograd < p_grad - (M - 1) * B * H * N * d * 4, (p_nograd, p_grad)


def test_cuda_graph_mixture_forward_backward():
    """The fused mixture's forward + backward capture in one CUDA graph and replay with fresh inputs."""
    B, H, N, d, M = 2, 3, 1000, 64, 3
    static = [t.clone().requires_grad_() for t in _mix_inputs(1, B, H, N, d, M)]
    dO = _upstream(1, (B, H, N, d))
    fn = _mix_fn(mixture_attention_triton, 64)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):  # compile and warm up outside the capture
            for t in static:
                t.grad = None
            fn(*static).backward(dO)
    torch.cuda.current_stream().wait_stream(s)
    for t in static:
        t.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*static).backward(dO)
    for seed in (2, 3):
        fresh = _mix_inputs(seed, B, H, N, d, M)
        with torch.no_grad():
            for dst, src in zip(static, fresh):
                dst.copy_(src)
        graph.replay()
        _assert_grads([t.grad for t in static], _grads(_mix_fn(_autograd_mixture, 64), fresh, dO),
                      names=_mix_names(M))


def test_mixture_double_backward_raises_and_validation():
    inputs = [t.requires_grad_() for t in _mix_inputs(3, 1, 2, 40, 16, 2)]
    out = _mix_fn(mixture_attention_triton, 16)(*inputs)
    (gg,) = torch.autograd.grad(out.sum(), inputs[0], create_graph=True)
    with pytest.raises(RuntimeError):
        gg.sum().backward()
    gates, V, *flat = (t.detach() for t in inputs)
    comps = [flat[:3], flat[3:]]
    with pytest.raises(ValueError):
        mixture_attention_triton(gates[..., :1], comps, V, 16)  # M = 1 gate for 2 components
    with pytest.raises(ValueError):
        mixture_attention_triton(gates[..., :0], [], V, 16)
    with pytest.raises(ValueError):
        mixture_attention_triton(gates, [flat[:3], flat[3:5]], V, 16)  # a component without b
    with pytest.raises(ValueError):
        mixture_attention_triton(gates, comps, V, 24)


def test_grads_finite_differences():
    """fp32 central differences of L = <dO, out> through the Triton forward: along random directions of every input
    (h = 1e-2) and for single coordinates, against the Triton backward. The ranks lie on a half-integer lattice
    (|r(y) - a(x)| >= 0.5), so no pair changes branch within the step: the order scores have a kink (the min) there,
    and with continuous random ranks a step along all of f flips about a dozen pairs (3% off at h = 1e-2)."""
    B, H, N, d, chunk = 1, 2, 70, 16, 16  # 5 chunks: intra tiles and a 3-level tree
    f, g, b, V = _inputs(21, B, H, N, d)
    gen = torch.Generator().manual_seed(4)
    f, g = f.clone(), g.clone()
    f[..., 1] = f[..., 0] - (torch.randint(-3, 4, (B, H, N), generator=gen) + 0.25).to(DEVICE)  # a(x) in Z + 1/4
    g[..., 1] = g[..., 0] - (torch.randint(-3, 4, (B, H, N), generator=gen) - 0.25).to(DEVICE)  # r(y) in Z - 1/4
    dO = _upstream(21, (B, H, N, d))
    xs = [f, g, b, V]
    grads = _grads(lambda *a: order_attention_triton(*a, chunk), xs, dO)
    loss = lambda args: float((order_attention_triton(*args, chunk).double() * dO.double()).sum())  # noqa: E731
    gen = torch.Generator().manual_seed(5)
    h = 1e-2
    for i, (x, gr) in enumerate(zip(xs, grads)):
        dirs = [torch.randn(x.shape, generator=gen).to(DEVICE) for _ in range(2)]
        for _ in range(3):  # single coordinates
            u = torch.zeros(x.numel(), device=DEVICE)
            u[int(torch.randint(0, x.numel(), (1,), generator=gen))] = 1.0
            dirs.append(u.view(x.shape))
        for u in dirs:
            plus, minus = list(xs), list(xs)
            plus[i], minus[i] = x + h * u, x - h * u
            fd = (loss(plus) - loss(minus)) / (2 * h)
            an = float((gr.double() * u.double()).sum())
            assert abs(fd - an) <= 2e-3 * max(1.0, abs(an)), (i, fd, an)


def test_cuda_graph_forward_backward():
    """Forward + backward capture in one CUDA graph (no host sync or host-to-device copy in either pass) and replay
    with fresh inputs."""
    shape = (2, 3, 1000, 64)
    static = [t.clone().requires_grad_() for t in _inputs(1, *shape)]
    dO = _upstream(1, shape)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):  # compile and warm up outside the capture
            for t in static:
                t.grad = None
            order_attention_triton(*static, 64).backward(dO)
    torch.cuda.current_stream().wait_stream(s)
    for t in static:
        t.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        order_attention_triton(*static, 64).backward(dO)
    for seed in (2, 3):
        fresh = _inputs(seed, *shape)
        with torch.no_grad():
            for dst, src in zip(static, fresh):
                dst.copy_(src)
        graph.replay()
        _assert_grads([t.grad for t in static], _grads(_rank_dense, fresh, dO))
