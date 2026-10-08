"""Tests of the long-context replacement experiment (experiments/attn_order_dim/longctx.py) on a tiny random Qwen3:
relative-position students are shift invariant, the dense training formula matches the sub-quadratic path (and the
sink-augmented SDPA of the rope128s control), the teacher matches the model's own attention, replaced models agree
between order_attention and dense_order_attention, recall counts are chunk invariant, and a few training steps reduce
the KL.

Run from the repo root: python -m pytest tests/test_longctx.py -q
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "attn_order_dim"))

from transformers import AutoModelForCausalLM, Qwen3Config  # noqa: E402

import longctx  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, D_MODEL, HEAD_DIM = 4, 32, 8


def _model(attn="sdpa", seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=101, hidden_size=D_MODEL, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=H, num_key_value_heads=2, head_dim=HEAD_DIM, max_position_embeddings=8192,
                      rope_theta=1e6)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation=attn, dtype=torch.float32).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _student(scorer, seed=0, max_slope=0.05):
    torch.manual_seed(seed)
    st = longctx.Student(D_MODEL, H, scorer, hidden=16, rope_dim=HEAD_DIM, max_slope=max_slope).to(DEVICE)
    if st.M:
        with torch.no_grad():  # nonzero slopes of both signs, so the position term matters
            st.slope.copy_(torch.randn(st.slope.shape, generator=torch.Generator().manual_seed(seed)) * 0.1)
    return st.eval()


def _cos_sin(model, pos):
    return model.model.rotary_emb(torch.zeros(1, device=DEVICE), pos[None])


@pytest.mark.parametrize("scorer", ["order1", "omix4", "rope128", "rope128s"])
@pytest.mark.parametrize("dense", [False, True])
def test_shift_invariance(scorer, dense):
    """Shifting every position by a constant leaves the student's attention (and its log-probs) unchanged."""
    model, st = _model(), _student(scorer)
    N = 50
    gen = torch.Generator().manual_seed(1)
    h = torch.randn(2, N, D_MODEL, generator=gen).to(DEVICE)
    V = torch.randn(2, H, N, HEAD_DIM, generator=gen).to(DEVICE)
    causal = torch.ones(N, N, dtype=torch.bool, device=DEVICE).tril()
    outs, lps = [], []
    for shift in (0, 777):
        pos = torch.arange(N, device=DEVICE) + shift
        out, cs = st(h, pos), _cos_sin(model, pos)
        outs.append(longctx.student_attend(st, out, V, cs, dense=dense))
        lps.append(longctx.student_log_probs(st, out, causal, cs))
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-4, atol=2e-4)
    torch.testing.assert_close(lps[0].exp(), lps[1].exp(), rtol=1e-4, atol=2e-4)
    if st.M:  # the slopes do change the attention: the test is not vacuous
        with torch.no_grad():
            st.slope.zero_()
        pos = torch.arange(N, device=DEVICE)
        flat = longctx.student_attend(st, st(h, pos), V, _cos_sin(model, pos), dense=dense)
        assert (flat - outs[0]).abs().max() > 1e-2


@pytest.mark.parametrize("scorer", ["order1", "omix4", "rope128", "rope128s"])
def test_dense_log_probs_match_attention_path(scorer):
    """exp(training log-probs) @ V equals the evaluation path (order_attention / mixture_attention / SDPA)."""
    model, st = _model(), _student(scorer, seed=3)
    N = 70
    gen = torch.Generator().manual_seed(2)
    h = torch.randn(1, N, D_MODEL, generator=gen).to(DEVICE)
    V = torch.randn(1, H, N, HEAD_DIM, generator=gen).to(DEVICE)
    pos = torch.arange(N, device=DEVICE)
    causal = torch.ones(N, N, dtype=torch.bool, device=DEVICE).tril()
    out, cs = st(h, pos), _cos_sin(model, pos)
    ref = longctx.student_log_probs(st, out, causal, cs).exp() @ V
    torch.testing.assert_close(longctx.student_attend(st, out, V, cs), ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(longctx.student_attend(st, out, V, cs, dense=True), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("D", [8, 128])
def test_sink_sdpa_scores_key0_by_sink_logit(D):
    """sink_sdpa = causal softmax with q k / sqrt(D) for keys >= 1 and b(x) for key 0; k(0) plays no role."""
    gen = torch.Generator().manual_seed(8)
    B, N = 2, 33
    q, k, V = (torch.randn(B, H, N, D, generator=gen).to(DEVICE) for _ in range(3))
    b = (torch.randn(B, H, N, generator=gen) * 3).to(DEVICE)
    s = q @ k.transpose(-1, -2) / D ** 0.5
    s[..., 0] = b
    causal = torch.ones(N, N, dtype=torch.bool, device=DEVICE).tril()
    ref = torch.softmax(s.masked_fill(~causal, float("-inf")), -1) @ V
    out = longctx.sink_sdpa(q, k, b, V)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    k2 = k.clone()
    k2[:, :, 0] += 5.0
    torch.testing.assert_close(longctx.sink_sdpa(q, k2, b, V), out)
    assert (longctx.sink_sdpa(q, k, b + 2.0, V) - out).abs().max() > 1e-2
    if DEVICE == "cuda":  # fp32 inputs must keep the memory-efficient kernel (the math fallback is quadratic in memory)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            torch.testing.assert_close(longctx.sink_sdpa(q, k, b, V), ref, rtol=1e-4, atol=1e-5)


def test_teacher_matches_model_attention():
    model = _model(attn="eager")
    ids = torch.randint(0, 101, (2, 40), generator=torch.Generator().manual_seed(0)).to(DEVICE)
    layers = [0, 2]
    with torch.no_grad(), longctx.capture(model, layers) as hs:
        attns = model(input_ids=ids, output_attentions=True, use_cache=False).attentions
    pos = torch.arange(40, device=DEVICE)
    causal = torch.ones(40, 40, dtype=torch.bool, device=DEVICE).tril()
    for l in layers:
        q, k, scale = longctx.teacher_qk(model, l, hs[l], pos)
        p = torch.softmax((q @ k.transpose(-1, -2) * scale).masked_fill(~causal, float("-inf")), -1)
        torch.testing.assert_close(p, attns[l], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("scorer", ["order1", "omix4"])
def test_replaced_model_fast_matches_dense(scorer):
    model = _model()
    students = {l: _student(scorer, seed=l) for l in (1, 2)}
    ids = torch.randint(0, 101, (1, 90), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    res = longctx.consistency(model, students, [1, 2], ids)
    assert res["max_abs_logit_diff"] < 1e-4
    with torch.no_grad():
        base = model(input_ids=ids, use_cache=False).logits
        with longctx.replaced(model, students, [1, 2]):
            rep = model(input_ids=ids, use_cache=False).logits
    assert (base - rep).abs().max() > 1e-3  # the replacement is in effect
    torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits, base)  # and undone afterwards


def test_perplexity_matches_full_logits():
    model = _model()
    tokens = torch.randint(0, 101, (200,), generator=torch.Generator().manual_seed(4))
    row = longctx.perplexity(model, tokens, 64, batch_tokens=128)
    win = tokens[:192].view(3, 64).to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids=win, use_cache=False).logits[:, :-1]
    nll = torch.nn.functional.cross_entropy(logits.flatten(0, 1), win[:, 1:].flatten())
    assert row["windows"] == 3 and row["predictions"] == 3 * 63
    assert abs(row["nll"] - nll.item()) < 1e-5


def test_recall_counts_chunk_invariant_and_match_dense():
    model, st = _model(), _student("order1", seed=5)
    N = 60
    h = torch.randn(1, N, D_MODEL, generator=torch.Generator().manual_seed(6)).to(DEVICE)
    pos = torch.arange(N, device=DEVICE)
    q, k, scale = longctx.teacher_qk(model, 1, h, pos)
    out = st(h, pos)
    ks = (1, 4, 16)
    full = longctx.recall_counts(q, k, scale, st, out, ks, q_chunk=N)
    chunked = longctx.recall_counts(q, k, scale, st, out, ks, q_chunk=7)
    for key in full:
        torch.testing.assert_close(full[key], chunked[key])
    # brute force from the dense training log-probs
    causal = torch.ones(N, N, dtype=torch.bool, device=DEVICE).tril()
    lq = longctx.student_log_probs(st, out, causal, None)[0, :, 1:]
    y = (q @ k.transpose(-1, -2) * scale).masked_fill(~causal, float("-inf"))[0, :, 1:].argmax(-1, keepdim=True)
    rank = (lq > lq.gather(-1, y)).sum(-1)
    assert full["queries"][0].item() == N - 1
    for kk in ks:
        torch.testing.assert_close(full[f"hit@{kk}"], (rank < kk).sum(-1).double())
        torch.testing.assert_close(full[f"nonsink_hit@{kk}"], ((rank < kk) & (y[..., 0] != 0)).sum(-1).double())


def test_training_reduces_kl():
    model = _model()
    tokens = torch.randint(0, 101, (2000,), generator=torch.Generator().manual_seed(7))
    scorers = ["order1", "omix4", "rope128", "rope128s"]
    students, stats = longctx.train_students(model, scorers, [1, 2], tokens, steps=60, t_train=32, batch=2, lr=3e-3,
                                             seed=0, warmup=5, hidden=32, log_every=10, log=lambda *a, **k: None)
    for sc, s in stats.items():
        first, last = s["curve"][0][1], s["curve"][-1][1]
        assert last < first, (sc, s["curve"])
        assert set(s["final_kl"]) == {"1", "2"}
    assert set(students["omix4"]) == {1, 2}


@pytest.mark.skipif(DEVICE != "cuda", reason="the Triton kernels need a GPU")
@pytest.mark.parametrize("scorer", ["order1", "omix4"])
def test_triton_attend_matches_torch_path_and_grads(scorer):
    """``impl='triton'`` gives the dense training formula's output and the same student-parameter gradients as the
    torch position tree (autograd through order_attention / mixture_attention)."""
    model, st = _model(), _student(scorer, seed=3)
    N = 150
    gen = torch.Generator().manual_seed(2)
    h = torch.randn(1, N, D_MODEL, generator=gen).to(DEVICE)
    V = torch.randn(1, H, N, HEAD_DIM, generator=gen).to(DEVICE)
    w = torch.randn(1, H, N, HEAD_DIM, generator=gen).to(DEVICE)
    pos = torch.arange(N, device=DEVICE)
    causal = torch.ones(N, N, dtype=torch.bool, device=DEVICE).tril()
    cs = _cos_sin(model, pos)
    with torch.no_grad():
        ref = longctx.student_log_probs(st, st(h, pos), causal, cs).exp() @ V
    grads = {}
    for impl in ("torch", "triton"):
        st.zero_grad(set_to_none=True)
        o = longctx.student_attend(st, st(h, pos), V, cs, impl=impl)
        torch.testing.assert_close(o.detach(), ref, rtol=1e-4, atol=1e-5)
        (o * w).sum().backward()
        grads[impl] = [p.grad.clone() for p in st.parameters()]
    for gt, gr in zip(grads["triton"], grads["torch"]):
        torch.testing.assert_close(gt, gr, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(DEVICE != "cuda", reason="the Triton kernels need a GPU")
@pytest.mark.parametrize("scorer", ["order1", "omix4"])
def test_replaced_model_triton_matches_torch(scorer):
    """The replaced model's logits through ``impl='triton'`` match ``impl='torch'`` and the dense path."""
    model = _model()
    students = {l: _student(scorer, seed=l) for l in (1, 2)}
    ids = torch.randint(0, 101, (1, 300), generator=torch.Generator().manual_seed(1)).to(DEVICE)
    res = longctx.consistency(model, students, [1, 2], ids, impl="triton")
    assert res["max_abs_logit_diff"] < 1e-4
    logits = {}
    with torch.no_grad():
        for impl in ("torch", "triton"):
            with longctx.replaced(model, students, [1, 2], impl=impl):
                logits[impl] = model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(logits["triton"], logits["torch"], rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(DEVICE != "cuda", reason="the Triton kernels need a GPU")
@pytest.mark.parametrize("M", [1, 4])
def test_bench_triton_impl(M):
    row = longctx.bench_order_attention(512, H=2, d=16, M=M, reps=1, impl="triton")
    assert row["finite"] and row["impl"] == "triton" and row["M"] == M
