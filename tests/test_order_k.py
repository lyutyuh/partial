"""Tests of the general-K range-tree aggregation (learning/order_k.py) against the O(N^2) score matrix.

Run from the repo root: python -m pytest tests/test_order_k.py -q
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.linear_order import arc_scores_quadratic, log_partition  # noqa: E402
from learning.order_k import arc_loss_k, decode_k, gold_arc_score_k, log_partition_k  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHAPES = [(1, 1), (2, 2), (3, 5), (4, 8), (3, 13), (2, 33)]


def _inputs(seed, B, L, K, scale=2.0, integer=False):
    gen = torch.Generator().manual_seed(seed)
    f = torch.randn(B, L, K, generator=gen) * scale
    g = torch.randn(B, L, K, generator=gen) * scale
    if integer:
        f, g = f.round(), g.round()
    lens = torch.randint(1, L + 1, (B,), generator=gen)
    lens[0] = L
    return f.to(DEVICE), g.to(DEVICE), lens.to(DEVICE)


def _valid(lens, L):
    return torch.arange(L, device=lens.device)[None, :] < lens[:, None]


def _z_quadratic(f, g, lens, root):
    z = arc_scores_quadratic(f, g, lens, root).logsumexp(-1)
    return torch.where(_valid(lens, f.shape[1]), z, 0.0)


@pytest.mark.parametrize("root", [True, False])
@pytest.mark.parametrize("K", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("B,L", SHAPES)
def test_log_partition_matches_quadratic(B, L, K, root):
    f, g, lens = _inputs(B * 100 + L, B, L, K)
    torch.testing.assert_close(log_partition_k(f, g, lens, root), _z_quadratic(f, g, lens, root), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("B,L", [(3, 13), (2, 33)])
def test_ties_integer_inputs(B, L, K):
    """Integer realizers make exact branch ties common; the partition must still cover every head exactly once."""
    f, g, lens = _inputs(7, B, L, K, integer=True)
    torch.testing.assert_close(log_partition_k(f, g, lens), _z_quadratic(f, g, lens, True), rtol=1e-5, atol=1e-5)
    s = arc_scores_quadratic(f, g, lens)
    _, scores = decode_k(f, g, lens)
    m = _valid(lens, L)
    torch.testing.assert_close(scores[m], s.amax(-1)[m])


@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("B,L", [(3, 5), (3, 13), (2, 33)])
def test_gradients_match_quadratic(B, L, K):
    f, g, lens = _inputs(11 + K, B, L, K)
    gz = torch.randn(B, L, generator=torch.Generator().manual_seed(3)).to(DEVICE)
    grads = []
    for fn in (log_partition_k, _z_quadratic):
        fr, gr = f.clone().requires_grad_(), g.clone().requires_grad_()
        (fn(fr, gr, lens, True) * gz).sum().backward()
        grads.append((fr.grad, gr.grad))
    torch.testing.assert_close(grads[0][0], grads[1][0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(grads[0][1], grads[1][1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("root", [True, False])
@pytest.mark.parametrize("K", [2, 3, 4, 5])
@pytest.mark.parametrize("B,L", SHAPES)
def test_decode_matches_quadratic_argmax(B, L, K, root):
    f, g, lens = _inputs(B + L + K, B, L, K)
    s = arc_scores_quadratic(f, g, lens, root)
    heads, scores = decode_k(f, g, lens, root)
    m = _valid(lens, L)
    assert torch.equal(heads[m].long(), s.argmax(-1)[m])
    torch.testing.assert_close(scores[m], s.amax(-1)[m], rtol=0, atol=1e-6)
    assert (heads[~m] == 0).all() and (scores[~m] == 0).all()


def test_k2_matches_triton_kernel():
    f, g, lens = _inputs(5, 4, 21, 2)
    torch.testing.assert_close(log_partition_k(f, g, lens), log_partition(f, g, lens), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("K", [2, 3])
def test_arc_loss_equals_cross_entropy(K):
    B, L = 4, 11
    f, g, lens = _inputs(21, B, L, K)
    gen = torch.Generator().manual_seed(1)
    heads = torch.stack([torch.randint(0, int(n) + 1, (L,), generator=gen) for n in lens.cpu()]).to(DEVICE)
    heads = torch.where(_valid(lens, L), heads, -1)
    fr, gr = f.clone().requires_grad_(), g.clone().requires_grad_()
    loss = arc_loss_k(fr, gr, heads, lens)
    loss.backward()
    fq, gq = f.clone().requires_grad_(), g.clone().requires_grad_()
    ref = torch.nn.functional.cross_entropy(arc_scores_quadratic(fq, gq, lens).movedim(-1, 1), heads, ignore_index=-1)
    ref.backward()
    torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fr.grad, fq.grad, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(gr.grad, gq.grad, rtol=1e-4, atol=1e-6)


def test_gold_score_reads_matrix():
    f, g, lens = _inputs(0, 3, 9, 3)
    heads = torch.randint(0, 10, (3, 9), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    s = arc_scores_quadratic(f, g, torch.full_like(lens, 9))
    torch.testing.assert_close(gold_arc_score_k(f, g, heads), s.gather(-1, heads.unsqueeze(-1)).squeeze(-1))


def test_large_magnitudes_stable():
    f, g, lens = _inputs(9, 2, 17, 3, scale=200.0)
    z = log_partition_k(f, g, lens)
    assert torch.isfinite(z).all()
    torch.testing.assert_close(z, _z_quadratic(f, g, lens, True), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("K", [1, 2, 3, 4])
@pytest.mark.parametrize("B,L", [(3, 5), (2, 33)])
def test_custom_backward_matches_autograd(B, L, K):
    f, g, lens = _inputs(31 + K, B, L, K)
    gz = torch.randn(B, L, generator=torch.Generator().manual_seed(4)).to(DEVICE)
    grads = []
    for ag in (True, False):
        fr, gr = f.clone().requires_grad_(), g.clone().requires_grad_()
        (log_partition_k(fr, gr, lens, True, autograd=ag) * gz).sum().backward()
        grads.append((fr.grad, gr.grad))
    torch.testing.assert_close(grads[0][0], grads[1][0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(grads[0][1], grads[1][1], rtol=1e-4, atol=1e-5)
