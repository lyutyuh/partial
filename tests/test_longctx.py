"""Tests of the long-context replacement experiment (experiments/attn_order_dim/longctx.py) on a tiny random Qwen3:
relative-position students are shift invariant, the dense training formula matches the sub-quadratic path (and the
sink-augmented SDPA of the rope128s control), the teacher matches the model's own attention, replaced models agree
between order_attention and dense_order_attention, recall counts are chunk invariant, and a few training steps reduce
the KL. Also the argmax-realizer fits of fit.py / report.py: the dot scorer is width-normalised (1 / sqrt(K)), the
temperature divides every scorer, the rope head-form control with a real layer's weights reproduces its attention, and
report.py selects variants, names its learned reference (never a ceiling) and flags dot train loss rising with width.

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

import fit  # noqa: E402
import longctx  # noqa: E402
import report  # noqa: E402

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
    xs = torch.arange(1, N, device=DEVICE)
    yy = y[..., 0]
    wrank = torch.where(yy == 0, 0, xs - yy + 1)  # sink first, then x, x-1, ...
    ns = yy != 0
    for kk in ks:
        torch.testing.assert_close(full[f"hit@{kk}"], (rank < kk).sum(-1).double())
        torch.testing.assert_close(full[f"nonsink_hit@{kk}"], ((rank < kk) & ns).sum(-1).double())
        # learning-free window baseline: hit iff the top key is the sink or among the kk - 1 most recent keys
        win = (yy == 0) | (yy >= xs - kk + 2)
        torch.testing.assert_close(full[f"win_hit@{kk}"], win.sum(-1).double())
        torch.testing.assert_close(full[f"nonsink_win_hit@{kk}"], (win & ns).sum(-1).double())
        union = (rank < kk // 2) | (wrank < kk - kk // 2)
        torch.testing.assert_close(full[f"nonsink_union_hit@{kk}"], (union & ns).sum(-1).double())
        nt = ns & (xs + 1 > kk)
        torch.testing.assert_close(full[f"nt_nonsink@{kk}"], nt.sum(-1).double())
        torch.testing.assert_close(full[f"nt_nonsink_hit@{kk}"], ((rank < kk) & nt).sum(-1).double())
        torch.testing.assert_close(full[f"nt_nonsink_win_hit@{kk}"], (win & nt).sum(-1).double())
    # the window baseline at kk = N covers every visible key
    big = longctx.recall_counts(q, k, scale, st, out, (N,), q_chunk=N)
    torch.testing.assert_close(big[f"win_hit@{N}"], big["queries"])


@pytest.mark.parametrize("scorer", ["order1", "omix4", "rope128s"])
def test_finetune_reduces_output_kl_and_freezes_base(scorer):
    """End-to-end fine-tuning lowers KL(base || replaced) and only moves the students' parameters."""
    model = _model()
    for prm in model.parameters():
        prm.requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    students = {l: _student(scorer, seed=l) for l in (1, 2)}
    tokens = torch.randint(0, 101, (4000,), generator=torch.Generator().manual_seed(9))
    impl = "triton" if DEVICE == "cuda" else "torch"
    stats = longctx.finetune_students(model, students, [1, 2], tokens, steps=40, t_ft=48, lr=3e-3, seed=0,
                                      warmup=4, impl=impl, logit_chunk=16, log_every=10, log=lambda *a, **k: None)
    first, last = stats["curve"][0][1], stats["curve"][-1][1]
    assert last < first, stats["curve"]
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert all(not st.training for st in students.values())


def test_finetune_loss_matches_full_vocab_kl():
    """The chunked lm_head loss equals the KL over the whole sequence computed in one piece."""
    model = _model()
    for prm in model.parameters():
        prm.requires_grad_(False)
    students = {l: _student("order1", seed=l) for l in (1, 2)}
    ids = torch.randint(0, 101, (1, 40), generator=torch.Generator().manual_seed(3)).to(DEVICE)
    with torch.no_grad():
        lt = torch.log_softmax(model(input_ids=ids).logits[:, :-1].float(), -1)
        with longctx.replaced(model, students, [1, 2], impl="torch"):
            ls = torch.log_softmax(model(input_ids=ids).logits[:, :-1].float(), -1)
    ref = (lt.exp() * (lt - ls)).sum(-1).mean().item()
    tokens = ids[0].cpu()
    stats = longctx.finetune_students(model, students, [1, 2], tokens, steps=1, t_ft=40, lr=0.0, seed=0, warmup=1,
                                      impl="torch", logit_chunk=7, log_every=1, log=lambda *a, **k: None)
    assert abs(stats["curve"][0][1] - ref) < 1e-5


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


# ---- argmax-realizer fits (fit.py) and their summary (report.py) ----

@pytest.mark.parametrize("K", [4, 128])
@pytest.mark.parametrize("masked", [True, False])
def test_fit_dot_scores_scaled_by_sqrt_width(K, masked):
    """fit.py's dot scorer is <f, g> / sqrt(K), as in replace.py and attention (it used to be unscaled)."""
    gen = torch.Generator().manual_seed(0)
    f, g = (torch.randn(2, 3, 20, K, generator=gen).to(DEVICE) for _ in range(2))
    ok = fit.allowed_mask(20, masked, DEVICE)
    ref = torch.log_softmax((f @ g.transpose(-1, -2) / K ** 0.5)[:, :, 1:].masked_fill(~ok, float("-inf")), -1)
    torch.testing.assert_close(fit.log_probs(f, g, None, "dot", masked), ref)
    assert fit.score_scale("dot", K) == pytest.approx(K ** -0.5)


def test_fit_dot_logit_scale_independent_of_width():
    """Regression: unscaled, a fresh dot realizer's logit spread grew like sqrt(K) (4x from K = 8 to 128), and the
    wide fits were badly conditioned: dot-128's train loss rose with width in late layers."""
    x = torch.randn(2, 30, D_MODEL, generator=torch.Generator().manual_seed(0)).to(DEVICE)
    pos = fit.sinusoid(30, fit.POS_DIM, DEVICE)
    raw, scaled = {}, {}
    for K in (8, 128):
        torch.manual_seed(0)
        with torch.no_grad():
            f, g, _ = fit.Realizer(D_MODEL, H, K).to(DEVICE)(x, pos)
        raw[K] = (f @ g.transpose(-1, -2)).std().item()
        scaled[K] = fit.scores(f, g, "dot").std().item()
    assert raw[128] / raw[8] > 3  # the test can see the problem
    assert 0.6 < scaled[128] / scaled[8] < 1.6


@pytest.mark.parametrize("scorer", ["order", "osum", "omix", "ogrp", "dot", "rope"])
def test_fit_temperature_divides_every_scorer(scorer):
    gen = torch.Generator().manual_seed(1)
    K = 6 if scorer != "ogrp" else 2
    w = fit.WIDTH[scorer](K)
    f, g = (torch.randn(2, 3, 12, w, generator=gen).to(DEVICE) for _ in range(2))
    gate = torch.randn(2, 3, 12, fit.GATES.get(scorer, lambda _: 0)(K), generator=gen).to(DEVICE)
    tau = 2.5
    got = fit.log_probs(f, g, gate, scorer, True, tau=tau)
    if scorer in ("dot", "rope"):
        ok = fit.allowed_mask(12, True, DEVICE)
        ref = torch.log_softmax((fit.scores(f, g, scorer) / tau)[:, :, 1:].masked_fill(~ok, float("-inf")), -1)
    else:  # order scores are positively homogeneous: s(f / tau, g / tau) = s(f, g) / tau
        ref = fit.log_probs(f / tau, g / tau, gate, scorer, True)
    torch.testing.assert_close(got, ref)
    assert (got - fit.log_probs(f, g, gate, scorer, True)).nan_to_num().abs().max() > 1e-2
    assert fit.score_scale(scorer, K, tau) == pytest.approx(1 / (tau * (w ** 0.5 if scorer in ("dot", "rope") else 1)))


def test_fit_rope_control_with_head_weights_reproduces_attention():
    """The rope control is the head's own function class: loaded with a Qwen3 layer's q/k projections and norms (the
    key projection repeated over its GQA group), its candidate distribution is the layer's attention renormalised over
    keys >= 1, so a rope-128 fit is a reference that can in principle reach 100%, unlike the dot-128 MLP realizer."""
    model = _model(attn="eager")
    ids = torch.randint(0, 101, (2, 40), generator=torch.Generator().manual_seed(0)).to(DEVICE)
    layer = 1
    with torch.no_grad(), longctx.capture(model, [layer]) as hs:
        attn = model(input_ids=ids, output_attentions=True, use_cache=False).attentions[layer]
    at = model.model.layers[layer].self_attn
    rr = fit.RopeRealizer(D_MODEL, H, HEAD_DIM, theta=model.config.rope_theta).to(DEVICE)
    with torch.no_grad():
        rr.q.weight.copy_(at.q_proj.weight)
        kv = at.k_proj.weight.view(-1, HEAD_DIM, D_MODEL)
        rr.k.weight.copy_(kv.repeat_interleave(H // kv.shape[0], dim=0).reshape(H * HEAD_DIM, D_MODEL))
        rr.q_norm.weight.copy_(at.q_norm.weight)
        rr.k_norm.weight.copy_(at.k_norm.weight)
        lp = fit.log_probs(*rr(hs[layer]), "rope", True)
    p = attn[:, :, 1:, 1:]
    p = p / p.sum(-1, keepdim=True)
    torch.testing.assert_close(lp.exp()[..., 1:], p, rtol=1e-4, atol=1e-6)
    assert torch.equal(lp.argmax(-1) - 1, p.argmax(-1))


def _synthetic_split(S, N=16, H_=2, d=D_MODEL, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(S, N, d, generator=gen)
    s = torch.einsum("snd,smd->snm", x, x) + torch.randn(S, N, N, generator=gen)
    s = s.masked_fill(~torch.ones(N, N, dtype=torch.bool).tril(), float("-inf"))
    tgt = torch.stack([s.argmax(-1), (torch.arange(N) - 1).clamp(min=0).expand(S, N)])[:H_]  # (H, S, N)
    return {"x": x.bfloat16().to(DEVICE), "tgt": tgt.short().to(DEVICE),
            "pmax": torch.rand(H_, S, N, generator=gen).half().to(DEVICE)}


@pytest.mark.skipif(DEVICE != "cuda", reason="fit.fit_one runs on the GPU")
@pytest.mark.parametrize("scorer,K", [("dot", 16), ("rope", 8), ("order", 2), ("omix", 2)])
def test_fit_one_held_out_split_and_temperature(scorer, K):
    full, test = _synthetic_split(10), _synthetic_split(4, seed=1)
    train, val = fit.split_val(full, 3)
    assert train["x"].shape[0] == 7 and val["tgt"].shape[1] == 3 and torch.equal(val["x"], full["x"][7:])
    pos = fit.sinusoid(16, fit.POS_DIM, "cuda")
    acc, loss = fit.fit_one(train, test, pos, scorer, K, True, steps=5, batch=2, lr=1e-3, seed=0, tau=2.0, val=val)
    assert {"acc_nonsink", "acc_conf", "n_nonsink", "val_acc_nonsink", "val_n_nonsink"} <= set(acc)
    assert len(acc["val_acc_nonsink"]) == 2 and loss == loss
    assert sum(acc["val_n_nonsink"]) == float((val["tgt"][:, :, 1:] != 0).sum())


def _fit_record(layer, scorer, K, acc, loss, **extra):
    return {"layer": layer, "scorer": scorer, "K": K, "masked": True, "train_loss": loss, "acc_nonsink": acc,
            "acc_conf": acc, "n_nonsink": [2000.0] * len(acc), "n_conf": [1000.0] * len(acc), **extra}


def _write(path, recs):
    import json
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return str(path)


def test_report_reference_variants_and_width_check(tmp_path, capsys):
    """report.py: (1) records from before the scaling fix (no score_scale: raw scores) whose dot train loss rises with
    width are flagged, and --strict fails; (2) a scaled dot-128 re-run is a second variant, the better one is used,
    and the flag clears; (3) the reference is the rope-128 control when fitted, and is never called a ceiling;
    (4) with held-out accuracies the variant is chosen on them, not on the test split."""
    heur = {"layer": 0, "scorer": "heuristics", "pmax_mean": [0.5, 0.5],
            "heuristics": {"sink": [0.1, 0.1], "prev": [0.6, 0.1], "self": [0.1, 0.1], "induction": [0.1, 0.1]}}
    legacy = _write(tmp_path / "qwen3-x_p0_old_1.jsonl", [
        heur, _fit_record(0, "dot", 16, [0.60, 0.60], 0.159), _fit_record(0, "dot", 128, [0.57, 0.57], 0.283),
        _fit_record(0, "osum", 16, [0.65, 0.65], 0.08)])
    assert report.main([legacy]) == 0 and report.main([legacy, "--strict"]) == 1
    out = capsys.readouterr().out
    assert "dot16 0.159 -> dot128 0.283" in out and "ceiling" not in out.replace("NOT a ceiling", "")
    assert "dot-128 MLP realizer" in out and "| osum | 16 | 32 | causal | 2 | 1 | 0.650 | 0.650 (100%, n=2)" in out

    scaled = dict(score_scale=fit.score_scale("dot", 128), tau=1.0, steps=1500, lr=1e-3)
    new = _write(tmp_path / "qwen3-x_p0_new_2.jsonl", [
        heur, _fit_record(0, "dot", 128, [0.69, 0.69], 0.031, **scaled),
        _fit_record(0, "rope", 128, [0.86, 0.86], 0.02, score_scale=fit.score_scale("rope", 128), steps=1500)])
    recs = report.load([legacy, new])["qwen3-x"]
    rec, n_var, split = report.select(recs, 1000)[(0, "dot", 128, True)]
    assert (rec["train_loss"], n_var, split) == (0.031, 2, "test")
    assert report.dot_width_violations(recs) == []
    assert report.main([legacy, new, "--strict"]) == 0
    out = capsys.readouterr().out
    assert "rope-128 head-form control" in out and "WARNING" not in out
    assert "| osum | 16 | 32 | causal | 2 | 1 | 0.650 | 0.650 (0%, n=2)" in out  # .65 / .86 < 0.9
    assert "| dot | 128 | 128 | causal | 2 | 2 (best on test) |" in out
    assert report.main([legacy, new, "--reference", "dot:128"]) == 0
    assert "| osum | 16 | 32 | causal | 2 | 1 | 0.650 | 0.650 (100%, n=2)" in capsys.readouterr().out  # .65 / .69

    # held-out accuracies decide when every variant has them, even against the test split
    val_old = _fit_record(0, "dot", 128, [0.57, 0.57], 0.283, val_acc_nonsink=[0.9, 0.9], val_n_nonsink=[500.0] * 2)
    val_new = _fit_record(0, "dot", 128, [0.69, 0.69], 0.031, val_acc_nonsink=[0.5, 0.5], val_n_nonsink=[500.0] * 2,
                          **scaled)
    recs = report.load([_write(tmp_path / "qwen3-y_p0_val_3.jsonl", [heur, val_old, val_new])])["qwen3-y"]
    rec, n_var, split = report.select(recs, 1000)[(0, "dot", 128, True)]
    assert (rec["train_loss"], n_var, split) == (0.283, 2, "val")
