"""Tests of the Triton linear-time kernels in learning/linear_order.py against the O(N^2) score matrix.

Without a GPU the kernels run under the Triton interpreter (TRITON_INTERPRET=1, set in conftest.py before torch loads),
so this file is a CPU test; on a GPU node the same tests exercise the compiled kernels.

Run from the repo root: python -m pytest tests/test_linear_order.py -q
"""
import os
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.linear_order import (  # noqa: E402
    arc_loss, arc_scores_quadratic, decode, gold_arc_score, log_partition, log_partition_torch,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHAPES = [(1, 1), (2, 2), (3, 7), (4, 16), (3, 37)]  # (batch, max length): odd and power-of-two lengths


@pytest.fixture(autouse=True)
def _quiet_log0():
    # log(0) = -inf is the intended "no head on this side" value; the interpreter warns about it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _inputs(seed, B, L, scale=3.0):
    gen = torch.Generator().manual_seed(seed)
    f = (torch.randn(B, L, 2, generator=gen) * scale).to(DEVICE)
    g = (torch.randn(B, L, 2, generator=gen) * scale).to(DEVICE)
    lens = torch.randint(1, L + 1, (B,), generator=gen)
    lens[0] = L  # always one full-length row
    return f, g, lens.to(DEVICE)


def _valid(lens, L):
    return torch.arange(L, device=lens.device)[None, :] < lens[:, None]


def _z_quadratic(f, g, lens, root):
    z = arc_scores_quadratic(f, g, lens, root).logsumexp(-1)
    return torch.where(_valid(lens, f.shape[1]), z, 0.0)


@pytest.mark.parametrize("root", [True, False])
@pytest.mark.parametrize("B,L", SHAPES)
@pytest.mark.parametrize("seed", range(3))
def test_log_partition_matches_quadratic(seed, B, L, root):
    f, g, lens = _inputs(seed, B, L)
    zq = _z_quadratic(f, g, lens, root)
    torch.testing.assert_close(log_partition(f, g, lens, root), zq, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(log_partition_torch(f, g, lens, root), zq, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("root", [True, False])
@pytest.mark.parametrize("B,L", SHAPES)
@pytest.mark.parametrize("seed", range(3))
def test_log_partition_gradients_match_quadratic(seed, B, L, root):
    f, g, lens = _inputs(seed, B, L)
    gz = torch.randn(B, L, generator=torch.Generator().manual_seed(seed + 100)).to(DEVICE)  # signed upstream grad
    grads = []
    for fn in (log_partition, _z_quadratic):
        fr, gr = f.clone().requires_grad_(), g.clone().requires_grad_()
        (fn(fr, gr, lens, root) * gz).sum().backward()
        grads.append((fr.grad, gr.grad))
    (df, dg), (df_ref, dg_ref) = grads
    torch.testing.assert_close(df, df_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(dg, dg_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("root", [True, False])
@pytest.mark.parametrize("B,L", SHAPES)
@pytest.mark.parametrize("seed", range(3))
def test_decode_matches_quadratic_argmax(seed, B, L, root):
    f, g, lens = _inputs(seed, B, L)
    s = arc_scores_quadratic(f, g, lens, root)
    heads, scores = decode(f, g, lens, root)
    m = _valid(lens, L)
    assert torch.equal(heads[m].long(), s.argmax(-1)[m])
    torch.testing.assert_close(scores[m], s.amax(-1)[m], rtol=0, atol=1e-6)
    assert (heads[~m] == 0).all() and (scores[~m] == 0).all()


@pytest.mark.parametrize("seed", range(3))
def test_arc_loss_equals_repo_cross_entropy(seed):
    """arc_loss == F.cross_entropy over the repo's (L, L + 1) score matrix (ROOT column of zeros), value and grads."""
    B, L = 4, 13
    f, g, lens = _inputs(seed, B, L)
    gen = torch.Generator().manual_seed(seed)
    heads = torch.stack([torch.randint(0, int(n) + 1, (L,), generator=gen) for n in lens.cpu()]).to(DEVICE)
    heads = torch.where(_valid(lens, L), heads, -1)

    losses, grads = [], []
    for use_kernel in (True, False):
        fr, gr = f.clone().requires_grad_(), g.clone().requires_grad_()
        if use_kernel:
            loss = arc_loss(fr, gr, heads, lens)
        else:
            s = arc_scores_quadratic(fr, gr, lens)
            loss = torch.nn.functional.cross_entropy(s.movedim(-1, 1), heads, ignore_index=-1)
        loss.backward()
        losses.append(loss.detach())
        grads.append((fr.grad, gr.grad))
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-6)
    for a, b in zip(grads[0], grads[1]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_gold_arc_score_reads_the_score_matrix():
    f, g, lens = _inputs(0, 3, 9)
    heads = torch.randint(0, 10, (3, 9), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    s = arc_scores_quadratic(f, g, torch.full_like(lens, 9))
    torch.testing.assert_close(gold_arc_score(f, g, heads), s.gather(-1, heads.unsqueeze(-1)).squeeze(-1))


@pytest.mark.parametrize("scale", [30.0, 300.0])
def test_large_magnitudes_are_stable(scale):
    """Realizer values far outside exp's range: log-space scans must stay finite and exact."""
    B, L = 3, 24
    f, g, lens = _inputs(7, B, L, scale=scale)
    zq = _z_quadratic(f, g, lens, True)
    z = log_partition(f, g, lens)
    assert torch.isfinite(z).all()
    torch.testing.assert_close(z, zq, rtol=1e-5, atol=1e-4)
    fr, gr = f.clone().requires_grad_(), g.clone().requires_grad_()
    log_partition(fr, gr, lens).sum().backward()
    assert torch.isfinite(fr.grad).all() and torch.isfinite(gr.grad).all()


def test_bf16_inputs_return_bf16_grads():
    f, g, lens = _inputs(3, 2, 10)
    fb, gb = f.bfloat16().requires_grad_(), g.bfloat16().requires_grad_()
    z = log_partition(fb, gb, lens)
    torch.testing.assert_close(z, _z_quadratic(fb, gb, lens, True), rtol=1e-5, atol=1e-5)
    z.sum().backward()
    assert fb.grad.dtype == torch.bfloat16 and gb.grad.dtype == torch.bfloat16


def test_ties_between_head_and_dependent_keys():
    """Integer-valued realizers make g1 - g2 == f1 - f2 common; either branch of F is then exact."""
    gen = torch.Generator().manual_seed(5)
    f = torch.randint(-3, 4, (2, 15, 2), generator=gen).float().to(DEVICE)
    g = torch.randint(-3, 4, (2, 15, 2), generator=gen).float().to(DEVICE)
    lens = torch.tensor([15, 9], device=DEVICE)
    torch.testing.assert_close(log_partition(f, g, lens), _z_quadratic(f, g, lens, True), rtol=1e-5, atol=1e-5)
    s = arc_scores_quadratic(f, g, lens)
    _, scores = decode(f, g, lens)
    m = _valid(lens, 15)
    torch.testing.assert_close(scores[m], s.amax(-1)[m])  # heads may differ on ties; the max score may not


def test_arc_loss_tie_gradient_matches_quadratic():
    """Exact F-tie at the gold arc: p(gold) = 1, so the loss and all gradients are 0 (review finding 1)."""
    f = torch.tensor([[[1.0, 1.0]]], device=DEVICE, requires_grad=True)
    g = torch.tensor([[[0.0, 0.0]]], device=DEVICE, requires_grad=True)
    loss = arc_loss(f, g, torch.tensor([[1]], device=DEVICE), torch.tensor([1], device=DEVICE), include_root=False)
    loss.backward()
    assert abs(loss.item()) < 1e-6
    assert f.grad.abs().max() < 1e-6 and g.grad.abs().max() < 1e-6


def test_negative_zero_key_keeps_heads_first():
    """Dependent key -0.0 vs head key +0.0 must still use branch 1 (review finding 2)."""
    f = torch.tensor([[[-0.0, 0.0]]], device=DEVICE, requires_grad=True)
    g = torch.tensor([[[0.0, 0.0]]], device=DEVICE, requires_grad=True)
    log_partition(f, g, torch.tensor([1], device=DEVICE), include_root=False).sum().backward()
    torch.testing.assert_close(f.grad, torch.tensor([[[-1.0, 0.0]]], device=DEVICE))
    torch.testing.assert_close(g.grad, torch.tensor([[[1.0, 0.0]]], device=DEVICE))


def test_invalid_lengths_are_rejected():
    f, g, _ = _inputs(0, 2, 5)
    with pytest.raises(ValueError):
        log_partition(f, g, torch.tensor([5, 6], device=DEVICE))
