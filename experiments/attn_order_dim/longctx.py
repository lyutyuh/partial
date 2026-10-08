"""Long-context quality of order-attention head replacement in Qwen3: perplexity and recall@k at 512..32768 tokens.

Students (one per layer, all heads) read the layer's attention input h (output of ``input_layernorm``) through a content
MLP (d_model -> 1024 -> outputs) and see positions only through learned slopes: for every (head, component, order k)
a scalar alpha_k is added as f_k(x) += alpha_k pos(x) and g_k(y) += alpha_k pos(y), so every score g_k(y) - f_k(x)
depends on positions only through pos(y) - pos(x) (ALiBi-like; a K = 2 head with slopes of opposite signs is a tent
around a content-chosen offset). Scorers:
    order1   one K = 2 order head per head: f, g in R^2 and a sink logit b(x) for key 0;
    omix4    query-gated mixture of 4 K = 2 order heads (gates, f, g, b from the content MLP);
    rope128  control: 128-d q, k from the content MLP, rotated by the model's own RoPE, scaled 1/sqrt(128), standard
             causal softmax over keys 0..x (no separate sink); quadratic, run through SDPA;
    rope128s control with the order students' sink: as rope128, but key 0 is scored by a per-query logit b(x) from the
             content MLP instead of q(x) k(0) (run through SDPA with b as an extra q/k channel). This is the matched
             control: rope128's sink mass collapses past T_train (0.6B, layer 22, queries at 8k-32k: teacher 0.58,
             rope128 0.01, rope128s 0.08), so rope128 vs order1 partly measures the sink parameterisation.
The slopes are bounded, |alpha| <= --max-slope (default 0.015 nats/token = 491 nats over 32k positions): the chunk
scans of ``order_attention`` overflow to inf/NaN when 64 rank-adjacent keys span more than ~709 nats of logit.

Training is online: windows of --t-train tokens from the WikiText-103 validation stream go through the frozen fp32
base model (SDPA), forward hooks capture h at the student layers, the teacher distribution of layer l is recomputed
from h with the layer's own q/k path (q_norm, k_norm, RoPE, repeat_kv, scaling, causal softmax), and every scorer's
students minimise KL(teacher || student) over queries x >= 1, summed over layers (one AdamW per scorer, all layers
jointly; the teacher is shared between scorers). Student dense log-probs use the formula of ``dense_order_attention``.

Evaluation uses the WikiText-103 test stream only: the first --eval-tokens (9 x 32768) tokens are cut into
non-overlapping windows of N tokens, so every N scores the same tokens; ppl = exp(mean NLL over all next-token
predictions); the tables also give the NLL gap to the base model at every N. Replaced layers keep v_proj / repeat_kv /
o_proj; order1 and omix4 run through the sub-quadratic ``order_attention`` / ``mixture_attention``, rope128(s) through
SDPA. Recall@k: fraction of queries whose teacher top key (argmax of the real head's causal attention, computed in
query chunks) is among the student's top-k keys.

Stages (``--stages``): train, ppl, recall, consistency (order_attention vs dense_order_attention logits), bench
(order_attention memory/time). Each run writes <out>/<tag>_part_<part>.json; ``--merge`` joins the parts into
<out>/<tag>.json and prints the tables.

Usage:
  python longctx.py --model Qwen/Qwen3-0.6B-Base --out <dir> --stages train --scorers order1 omix4 rope128 rope128s
  python longctx.py --model Qwen/Qwen3-0.6B-Base --out <dir> --stages ppl consistency --scorers order1 omix4 rope128s
  python longctx.py --model Qwen/Qwen3-0.6B-Base --out <dir> --stages recall bench
  python longctx.py --model Qwen/Qwen3-0.6B-Base --out <dir> --merge
"""
import argparse
import contextlib
import glob
import json
import math
import os
import sys
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from learning.order_attention import dense_order_attention, mixture_attention, order_attention  # noqa: E402

WIKI = "Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/{split}-00000-of-00001.parquet"
NEG = float("-inf")
COMPONENTS = {"order1": 1, "omix4": 4, "rope128": 0, "rope128s": 0}  # K = 2 order components per head; 0 = dot product
DOT_SINK = {"rope128s"}  # dot-product controls whose key 0 is scored by a per-query sink logit b(x), as in order heads
REPLACED = list(range(22, 28))
RECALL_ONLY = [2, 8, 14]
SLOPE_GAIN = 10.0  # alpha = max_slope * tanh(SLOPE_GAIN * raw): raw moves ~lr per Adam step, so this speeds the slopes
LOGIT_CHUNK = 4096  # positions per lm_head chunk in the NLL (a 32k x 151936 fp32 logit matrix would be 20 GB)


# ----------------------------------------------------------------------------------------------------------- students
class Student(nn.Module):
    """Per-layer student for all heads: content MLP on the attention input + bounded relative position slopes.

    Args:
        d_in: width of the attention input h.
        heads: number of query heads.
        scorer: 'order1', 'omix4', 'rope128' or 'rope128s'.
        hidden: hidden width of the content MLP.
        rope_dim: q/k width of 'rope128(s)' (must equal the model's head_dim, whose RoPE tables are reused).
        max_slope: bound on |alpha| in nats per token.
    """

    def __init__(self, d_in, heads, scorer, hidden=1024, rope_dim=128, max_slope=0.015):
        super().__init__()
        self.d_in, self.heads, self.scorer, self.hidden = d_in, heads, scorer, hidden
        self.rope_dim, self.max_slope = rope_dim, max_slope
        self.M, self.sink = COMPONENTS[scorer], scorer in DOT_SINK
        per_head = 2 * rope_dim + self.sink if self.M == 0 else 5 * self.M + (self.M if self.M > 1 else 0)
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, heads * per_head))
        if self.M:
            self.slope = nn.Parameter(torch.zeros(heads, self.M, 2))

    def config(self):
        return {"d_in": self.d_in, "heads": self.heads, "scorer": self.scorer, "hidden": self.hidden,
                "rope_dim": self.rope_dim, "max_slope": self.max_slope}

    def alpha(self):
        """(H, M, 2) slopes in nats per token."""
        return self.max_slope * torch.tanh(SLOPE_GAIN * self.slope)

    def forward(self, h, pos):
        """h: (B, n, d_in); pos: (n,) positions. Returns q, k (B, H, n, rope_dim) for 'rope128' (plus the sink
        logit b (B, H, n) for 'rope128s'), otherwise f, g (B, H, n, M, 2) with the slopes folded in, b (B, H, n, M)
        sink logits and gates (B, H, n, M) or None."""
        B, n, _ = h.shape
        out = self.net(h.float()).view(B, n, self.heads, -1).transpose(1, 2)
        if self.M == 0:
            r = self.rope_dim
            qk = {"q": out[..., :r], "k": out[..., r:2 * r]}
            return {**qk, "b": out[..., 2 * r]} if self.sink else qk
        M = self.M
        fg = out[..., :4 * M].reshape(B, self.heads, n, M, 2, 2)
        shift = self.alpha()[None, :, None] * pos.float()[None, None, :, None, None]  # (1, H, n, M, 2)
        return {"f": fg[..., 0, :] + shift, "g": fg[..., 1, :] + shift, "b": out[..., 4 * M:5 * M],
                "gates": out[..., 5 * M:] if M > 1 else None}


def components(out):
    """[(f, g, b), ...] per order component, in the (B, H, n, 2) / (B, H, n) layout of ``order_attention``."""
    return [(out["f"][..., m, :].contiguous(), out["g"][..., m, :].contiguous(), out["b"][..., m].contiguous())
            for m in range(out["f"].shape[-2])]


def order_scores(f, g, b):
    """Scores of one K = 2 order component, (B, H, nq, nk): min_k g_k(y) - f_k(x), key 0 scored by b(x)."""
    s = torch.minimum(g[..., None, :, 0] - f[..., :, None, 0], g[..., None, :, 1] - f[..., :, None, 1])
    return torch.cat([b.unsqueeze(-1), s[..., 1:]], dim=-1)


def sink_sdpa(q, k, b, V):
    """Causal softmax attention with scores q(x) k(y) / sqrt(D) for keys y >= 1 and b(x) for key 0, through SDPA.

    b enters as an extra q/k channel (q' = [q / sqrt(D), b], k'(0) = [0, 1], k'(y) = [k(y), 0]); the channels are
    zero-padded to a multiple of 8 so that fp32 inputs keep the memory-efficient kernel (an odd width falls back to
    the quadratic-memory math backend).
    """
    D = q.shape[-1]
    pad = -(D + 1) % 8
    first = torch.zeros(k.shape[-2], device=k.device, dtype=k.dtype)
    first[0] = 1.0
    qa = torch.cat([q / math.sqrt(D), b[..., None], q.new_zeros(*q.shape[:-1], pad)], dim=-1)
    ka = torch.cat([k * (1.0 - first)[:, None], first[:, None].expand(*k.shape[:-1], 1),
                    k.new_zeros(*k.shape[:-1], pad)], dim=-1)
    return F.scaled_dot_product_attention(qa, ka, V, is_causal=True, scale=1.0)


def student_log_probs(student, out, causal, cos_sin):
    """Dense log p(y | x), (B, H, n, n), -inf outside the causal mask ``causal`` (n, n)."""
    if student.M == 0:
        q, k = apply_rotary_pos_emb(out["q"], out["k"], *cos_sin)
        s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        if "b" in out:  # rope128s: key 0 is scored by the sink logit
            s = torch.cat([out["b"].unsqueeze(-1), s[..., 1:]], dim=-1)
        return torch.log_softmax(s.masked_fill(~causal, NEG), dim=-1)
    lps = [torch.log_softmax(order_scores(*c).masked_fill(~causal, NEG), dim=-1) for c in components(out)]
    if student.M == 1:
        return lps[0]
    lw = torch.log_softmax(out["gates"], dim=-1)
    # finite fill outside the mask: logsumexp over an all -inf slice has a NaN gradient
    mix = torch.stack([lp + lw[..., m, None] for m, lp in enumerate(lps)], dim=-1)
    mix = mix.masked_fill(~causal[..., None], -1e30)
    return torch.logsumexp(mix, dim=-1).masked_fill(~causal, NEG)


def student_attend(student, out, V, cos_sin, dense=False):
    """Causal attention output (B, H, n, d) of the student; sub-quadratic for order scorers unless ``dense``."""
    if student.M == 0:
        q, k = apply_rotary_pos_emb(out["q"], out["k"], *cos_sin)
        if "b" in out:
            return sink_sdpa(q, k, out["b"], V)
        return F.scaled_dot_product_attention(q, k, V, is_causal=True, scale=1.0 / math.sqrt(q.shape[-1]))
    comps = components(out)
    if dense:
        if student.M == 1:
            return dense_order_attention(*comps[0], V)
        w = torch.softmax(out["gates"].float(), dim=-1)
        return sum(w[..., m:m + 1] * dense_order_attention(*c, V) for m, c in enumerate(comps))
    if student.M == 1:
        return order_attention(*comps[0], V, causal=True)
    return mixture_attention(out["gates"], comps, V, causal=True)


def save_students(students, path):
    blob = {"layers": {l: {"config": st.config(), "state": st.state_dict()} for l, st in students.items()}}
    torch.save(blob, path + ".tmp")
    os.replace(path + ".tmp", path)  # atomic: other processes poll for this file


def load_students(path, device):
    blob = torch.load(path, map_location=device)
    students = {}
    for l, rec in blob["layers"].items():
        st = Student(**rec["config"]).to(device)
        st.load_state_dict(rec["state"])
        students[int(l)] = st.eval()
    return students


# ------------------------------------------------------------------------------------------------- teacher and model
def teacher_qk(model, layer, h, pos):
    """Rotated, repeated q, k (B, H, n, D) of the real head and its scaling, recomputed from the attention input."""
    at = model.model.layers[layer].self_attn
    B, n, _ = h.shape
    cos, sin = model.model.rotary_emb(h, pos[None].expand(B, -1))
    q = at.q_norm(at.q_proj(h).view(B, n, -1, at.head_dim)).transpose(1, 2)
    k = at.k_norm(at.k_proj(h).view(B, n, -1, at.head_dim)).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, repeat_kv(k, at.num_key_value_groups), at.scaling


@contextlib.contextmanager
def capture(model, layers):
    """Collects {layer: attention input} (outputs of input_layernorm) during a forward pass."""
    store = {}
    hooks = [model.model.layers[l].input_layernorm.register_forward_hook(
        lambda mod, inp, out, l=l: store.__setitem__(l, out.detach())) for l in layers]
    try:
        yield store
    finally:
        for hk in hooks:
            hk.remove()


class ReplacedAttention(nn.Module):
    """Qwen3 attention whose weights come from a student for all heads; v_proj, repeat_kv and o_proj are the layer's."""

    def __init__(self, attn, student, dense=False):
        super().__init__()
        self.attn, self.student, self.dense = attn, student, dense

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None, **kwargs):
        at = self.attn
        B, n, _ = hidden_states.shape
        v = at.v_proj(hidden_states).view(B, n, -1, at.head_dim).transpose(1, 2)
        v = repeat_kv(v, at.num_key_value_groups)
        pos = position_ids[0] if position_ids is not None else torch.arange(n, device=hidden_states.device)
        out = self.student(hidden_states, pos)
        o = student_attend(self.student, out, v.float(), position_embeddings, self.dense)
        return at.o_proj(o.to(v.dtype).transpose(1, 2).reshape(B, n, -1)), None


@contextlib.contextmanager
def replaced(model, students, layers, dense=False):
    """Temporarily replaces the attention of ``layers`` with ``students[l]``."""
    orig = {l: model.model.layers[l].self_attn for l in layers}
    try:
        for l in layers:
            model.model.layers[l].self_attn = ReplacedAttention(orig[l], students[l], dense)
        yield
    finally:
        for l in layers:
            model.model.layers[l].self_attn = orig[l]


def load_stream(tokenizer, split):
    """WikiText-103 raw split as one token stream without special tokens."""
    path = glob.glob(os.path.join(os.environ["HF_HUB_CACHE"], "datasets--" + WIKI.format(split=split)))[0]
    text = "".join(pq.read_table(path).column("text").to_pylist())
    return torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=torch.long)


# ----------------------------------------------------------------------------------------------------------- training
def train_students(model, scorers, layers, tokens, steps, t_train, batch, lr, seed, max_slope=0.015, warmup=100,
                   slope_lr_mult=1.0, hidden=1024, log_every=50, log=print):
    """Fits every scorer's students on random windows of ``tokens`` (KL(teacher || student), summed over layers).

    Returns ({scorer: {layer: Student}}, {scorer: stats}).
    """
    device = next(model.parameters()).device
    cfg = model.config
    torch.manual_seed(seed)
    students, opts, scheds = {}, {}, {}
    for sc in scorers:
        students[sc] = {l: Student(cfg.hidden_size, cfg.num_attention_heads, sc, hidden, cfg.head_dim,
                                   max_slope).to(device).train() for l in layers}
        net = [p for st in students[sc].values() for p in st.net.parameters()]
        slopes = [st.slope for st in students[sc].values() if st.M]
        groups = [{"params": net, "lr": lr}] + ([{"params": slopes, "lr": lr * slope_lr_mult}] if slopes else [])
        opts[sc] = torch.optim.AdamW(groups, weight_decay=0.0)
        scheds[sc] = torch.optim.lr_scheduler.LambdaLR(opts[sc], lambda t: min(1.0, (t + 1) / warmup) * 0.5 * (
            1 + math.cos(math.pi * min(t, steps) / steps)))
    gen = torch.Generator().manual_seed(seed)
    pos = torch.arange(t_train, device=device)
    causal = torch.ones(t_train, t_train, dtype=torch.bool, device=device).tril()
    cos_sin = model.model.rotary_emb(torch.zeros(1, device=device), pos[None])
    curve = {sc: [] for sc in scorers}
    tail = {sc: {l: [] for l in layers} for sc in scorers}
    acc = {sc: torch.zeros(len(layers), device=device) for sc in scorers}
    t0 = time.time()
    for step in range(steps):
        starts = torch.randint(0, tokens.shape[0] - t_train + 1, (batch,), generator=gen)
        ids = torch.stack([tokens[s:s + t_train] for s in starts.tolist()]).to(device)
        with torch.no_grad(), capture(model, layers) as hs:
            model.model(input_ids=ids, use_cache=False)
        for i, l in enumerate(layers):
            h = hs.pop(l)
            with torch.no_grad():
                q, k, scale = teacher_qk(model, l, h, pos)
                lt = (q @ k.transpose(-1, -2) * scale).masked_fill(~causal, NEG)
                p = torch.softmax(lt, dim=-1)
                neg_ent = (p * torch.log_softmax(lt, dim=-1).masked_fill(~causal, 0.0)).sum(-1)  # (B, H, n)
                del q, k, lt
            for sc in scorers:
                st = students[sc][l]
                lq = student_log_probs(st, st(h, pos), causal, cos_sin)
                kl = neg_ent - (p * lq.masked_fill(~causal, 0.0)).sum(-1)  # KL(teacher || student) per query
                loss = kl[:, :, 1:].mean()
                loss.backward()
                acc[sc][i] += loss.detach()
                del lq, kl, loss
        for sc in scorers:
            opts[sc].step()
            scheds[sc].step()
            opts[sc].zero_grad(set_to_none=True)
        if (step + 1) % log_every == 0 or step + 1 == steps:
            n_int = (step % log_every) + 1
            msg = []
            for sc in scorers:
                per_layer = (acc[sc] / n_int).tolist()
                curve[sc].append([step + 1, sum(per_layer)])
                if step + 1 > steps - max(log_every, 50):
                    for l, v in zip(layers, per_layer):
                        tail[sc][l].append(v)
                msg.append(f"{sc} {sum(per_layer):.3f}")
                acc[sc].zero_()
            log(f"  step {step + 1}/{steps} ({time.time() - t0:.0f}s): KL summed over {len(layers)} layers: "
                + ", ".join(msg), flush=True)
    stats = {}
    for sc in scorers:
        st_stats = {"steps": steps, "t_train": t_train, "batch": batch, "seconds": round(time.time() - t0, 1),
                    "curve": curve[sc], "final_kl": {str(l): sum(v) / max(len(v), 1) for l, v in tail[sc].items()}}
        if COMPONENTS[sc]:
            al = torch.stack([students[sc][l].alpha().detach().abs().flatten() for l in layers])
            st_stats["slope_abs_mean"] = al.mean(1).tolist()
            st_stats["slope_frac_at_bound"] = (al > 0.9 * max_slope).float().mean(1).tolist()
        stats[sc] = st_stats
    for sc in scorers:
        for st in students[sc].values():
            st.eval()
    return students, stats


# --------------------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def window_nll(model, ids):
    """Summed next-token NLL of a batch of windows (B, N); the lm_head runs on chunks of LOGIT_CHUNK predictions."""
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1].flatten(0, 1)
    tgt = ids[:, 1:].flatten()
    total = 0.0
    for j in range(0, tgt.shape[0], LOGIT_CHUNK):
        logits = model.lm_head(h[j:j + LOGIT_CHUNK]).float()
        total += F.cross_entropy(logits, tgt[j:j + LOGIT_CHUNK], reduction="sum").item()
    return total


@torch.no_grad()
def perplexity(model, tokens, N, max_windows=0, batch_tokens=32768):
    """ppl over non-overlapping N-token windows of ``tokens``; returns a result row (ppl, NLL, time, peak memory)."""
    device = next(model.parameters()).device
    win = tokens[: (tokens.shape[0] // N) * N].view(-1, N)
    if max_windows:
        win = win[:max_windows]
    bs = max(1, batch_tokens // N)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total = 0.0
    for i in range(0, win.shape[0], bs):
        total += window_nll(model, win[i:i + bs].to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    count = win.shape[0] * (N - 1)
    row = {"N": N, "windows": win.shape[0], "predictions": count, "nll": total / count,
           "ppl": math.exp(total / count) if math.isfinite(total) else float("nan"),
           "seconds": round(time.time() - t0, 2)}
    if device.type == "cuda":
        row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    return row


@torch.no_grad()
def recall_counts(q, k, scale, student, out, ks, q_chunk=512):
    """Per-head counts for one window: queries x >= 1, hits@k, non-sink queries, non-sink hits@k, summed KL.

    The teacher logits (q k^T scale) and the student's order scores are formed for ``q_chunk`` queries at a time
    against all N keys (causal), so memory is O(H q_chunk N). Ties count as hits (strictly higher keys are counted).
    """
    H, N = q.shape[1], q.shape[2]
    dev = q.device
    keys = torch.arange(N, device=dev)
    zeros = lambda: torch.zeros(H, device=dev, dtype=torch.float64)  # noqa: E731
    c = {"queries": zeros(), "nonsink": zeros(), "kl": zeros(), "sink_mass": zeros()}
    for kk in ks:
        c[f"hit@{kk}"], c[f"nonsink_hit@{kk}"] = zeros(), zeros()
    comps = components(out) if student.M else None
    for x0 in range(0, N, q_chunk):
        x1 = min(N, x0 + q_chunk)
        xs = torch.arange(x0, x1, device=dev)
        vis = keys[None, :] <= xs[:, None]  # (qc, N)
        lt = (q[:, :, x0:x1] @ k.transpose(-1, -2) * scale).masked_fill(~vis, NEG)  # (B, H, qc, N)
        if student.M == 1:
            f, g, b = comps[0]
            s = order_scores(f[:, :, x0:x1], g, b[:, :, x0:x1]).masked_fill(~vis, NEG)
        else:
            raise ValueError("recall is implemented for single-component order students (order1)")
        y = lt.argmax(-1, keepdim=True)  # teacher top key
        rank = (s > s.gather(-1, y)).sum(-1)  # (B, H, qc): keys the student scores strictly above it
        valid = (xs >= 1).to(torch.float64)[None, None]
        ns = (y[..., 0] != 0).to(torch.float64) * valid
        c["queries"] += valid.expand_as(ns).sum((0, 2))
        c["nonsink"] += ns.sum((0, 2))
        for kk in ks:
            hit = (rank < kk).to(torch.float64)
            c[f"hit@{kk}"] += (hit * valid).sum((0, 2))
            c[f"nonsink_hit@{kk}"] += (hit * ns).sum((0, 2))
        lpt, lps = torch.log_softmax(lt, -1), torch.log_softmax(s, -1)
        pt = lpt.exp()
        kl = torch.where(vis, pt * (lpt - lps), 0.0).sum(-1)
        c["kl"] += (kl.to(torch.float64) * valid).sum((0, 2))
        c["sink_mass"] += (pt[..., 0].to(torch.float64) * valid).sum((0, 2))
        del lt, s, lpt, lps, pt, kl
    return c


@torch.no_grad()
def recall_eval(model, students, layers, tokens, N, max_windows, ks, q_chunk=512, log=print):
    """Recall@k of the teacher's top key and KL per layer and head over ``max_windows`` windows of N tokens."""
    device = next(model.parameters()).device
    win = tokens[: (tokens.shape[0] // N) * N].view(-1, N)
    if max_windows:
        win = win[:max_windows]
    pos = torch.arange(N, device=device)
    tot = {}
    t0 = time.time()
    for w in range(win.shape[0]):
        with capture(model, layers) as hs:
            model.model(input_ids=win[w:w + 1].to(device), use_cache=False)
        for l in layers:
            h = hs.pop(l)
            q, k, scale = teacher_qk(model, l, h, pos)
            c = recall_counts(q, k, scale, students[l], students[l](h, pos), ks, q_chunk)
            if l not in tot:
                tot[l] = c
            else:
                for key in c:
                    tot[l][key] += c[key]
            del q, k, h
    rows = []
    for l in layers:
        c = tot[l]
        row = {"layer": l, "N": N, "windows": win.shape[0], "queries": int(c["queries"][0].item()),
               "nonsink_frac": (c["nonsink"] / c["queries"]).mean().item(),
               "kl": (c["kl"] / c["queries"]).mean().item(),
               "sink_mass": (c["sink_mass"] / c["queries"]).mean().item(), "seconds": round(time.time() - t0, 1),
               "per_head": {}}
        for kk in ks:
            row[f"recall@{kk}"] = (c[f"hit@{kk}"] / c["queries"]).mean().item()
            row[f"nonsink_recall@{kk}"] = (c[f"nonsink_hit@{kk}"] / c["nonsink"].clamp(min=1)).mean().item()
            row["per_head"][f"recall@{kk}"] = (c[f"hit@{kk}"] / c["queries"]).tolist()
            row["per_head"][f"nonsink_recall@{kk}"] = (c[f"nonsink_hit@{kk}"] / c["nonsink"].clamp(min=1)).tolist()
        row["per_head"]["kl"] = (c["kl"] / c["queries"]).tolist()
        rows.append(row)
    return rows


@torch.no_grad()
def consistency(model, students, layers, ids):
    """Max |logit difference| of the model with ``layers`` replaced through order_attention vs dense_order_attention."""
    outs = []
    for dense in (False, True):
        with replaced(model, students, layers, dense=dense):
            outs.append(model(input_ids=ids, use_cache=False).logits.float())
    nll = [F.cross_entropy(o[0, :-1], ids[0, 1:]).item() for o in outs]
    return {"max_abs_logit_diff": (outs[0] - outs[1]).abs().max().item(),
            "max_abs_logit": outs[1].abs().max().item(), "nll_fast": nll[0], "nll_dense": nll[1]}


@torch.no_grad()
def bench_order_attention(N, H=16, d=128, M=1, reps=3, device="cuda"):
    """Time and peak extra memory of one no-grad order_attention (M = 1) or mixture_attention (M > 1) forward."""
    gen = torch.Generator(device="cpu").manual_seed(N)
    rnd = lambda *s: (torch.randn(*s, generator=gen) * 2).to(device)  # noqa: E731
    pos = torch.arange(N, dtype=torch.float32, device=device)[:, None] * 0.01
    comps = [(rnd(1, H, N, 2) + pos, rnd(1, H, N, 2) + pos, rnd(1, H, N)) for _ in range(M)]
    V, gates = rnd(1, H, N, d), rnd(1, H, N, M)
    fn = (lambda: order_attention(*comps[0], V)) if M == 1 else (lambda: mixture_attention(gates, comps, V))
    fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    sec = (time.time() - t0) / reps
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    finite = bool(torch.isfinite(out).all())
    del out
    q = rnd(1, H, N, d)
    F.scaled_dot_product_attention(q, q, V, is_causal=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        F.scaled_dot_product_attention(q, q, V, is_causal=True)
    torch.cuda.synchronize()
    return {"N": N, "H": H, "d": d, "M": M, "seconds": round(sec, 4), "peak_extra_gb": round(peak, 3),
            "finite": finite, "sdpa_fp32_seconds": round((time.time() - t0) / reps, 4)}


# --------------------------------------------------------------------------------------------------------------- main
class Part:
    """Result file of one run; rewritten after every measurement so that partial results survive a time limit."""

    def __init__(self, path, model_name, args):
        self.path = path
        self.data = {"model": model_name, "args": {args.part: vars(args)}}

    def add(self, key, row):
        """Appends a result row to the list ``key``."""
        self.data.setdefault(key, []).append(row)
        self._write()

    def update(self, key, entries):
        """Merges ``entries`` into the dict ``key``."""
        self.data.setdefault(key, {}).update(entries)
        self._write()

    def _write(self):
        with open(self.path + ".tmp", "w") as fh:
            json.dump(self.data, fh, indent=1)
        os.replace(self.path + ".tmp", self.path)


def merge(out_dir, tag):
    res = {}
    for path in sorted(glob.glob(os.path.join(out_dir, f"{tag}_part_*.json"))):
        with open(path) as fh:
            part = json.load(fh)
        for key, value in part.items():
            if isinstance(value, list):
                res.setdefault(key, []).extend(value)
            elif isinstance(value, dict):
                res.setdefault(key, {}).update(value)
            else:
                res[key] = value
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print_tables(res)
    return res


def print_tables(res):
    print(f"== {res.get('model')}")
    rows = res.get("ppl", [])
    Ns = sorted({r["N"] for r in rows})
    if rows:
        print("perplexity (WikiText-103 test, same tokens at every N)")
        print(f"{'scorer':10s} {'layers':7s} " + " ".join(f"{n:>8d}" for n in Ns))
        keys = sorted({(r["scorer"], r["layers"]) for r in rows}, key=lambda t: (t[0] != "base", t[1] != "22-27", t))
        for sc, ls in keys:
            vals = {r["N"]: r["ppl"] for r in rows if (r["scorer"], r["layers"]) == (sc, ls)}
            print(f"{sc:10s} {ls:7s} " + " ".join(f"{vals[n]:8.3f}" if n in vals else f"{'-':>8s}" for n in Ns))
        base = {r["N"]: r["nll"] for r in rows if r["scorer"] == "base"}
        if base:
            print("NLL gap to the base model (nats per token, same windows)")
            print(f"{'scorer':10s} {'layers':7s} " + " ".join(f"{n:>8d}" for n in Ns))
            for sc, ls in keys:
                if sc != "base":
                    gap = {r["N"]: r["nll"] - base[r["N"]] for r in rows
                           if (r["scorer"], r["layers"]) == (sc, ls) and r["N"] in base}
                    print(f"{sc:10s} {ls:7s} " + " ".join(f"{gap[n]:+8.4f}" if n in gap else f"{'-':>8s}" for n in Ns))
    rows = res.get("recall", [])
    if rows:
        Ns = sorted({r["N"] for r in rows})
        for metric, what in (("recall@64", "teacher top key in the student's top 64"),
                             ("nonsink_recall@64", "same, queries whose teacher top key is not the sink"),
                             ("recall@8", "teacher top key in the student's top 8"),
                             ("kl", "KL(teacher || student) per query, nats")):
            print(f"{metric} of order1 students ({what})")
            print(f"{'layer':6s} " + " ".join(f"{n:>8d}" for n in Ns))
            for l in sorted({r["layer"] for r in rows}):
                vals = {r["N"]: r[metric] for r in rows if r["layer"] == l}
                print(f"{l:<6d} " + " ".join(f"{vals[n]:8.4f}" if n in vals else f"{'-':>8s}" for n in Ns))
    for r in res.get("consistency", []):
        print(f"consistency {r['scorer']}: max |logit diff| {r['max_abs_logit_diff']:.2e} "
              f"(max |logit| {r['max_abs_logit']:.1f}), NLL {r['nll_fast']:.5f} vs {r['nll_dense']:.5f}")
    for r in res.get("bench", []):
        print(f"bench order_attention N={r['N']} M={r['M']}: {r['seconds'] * 1e3:.1f} ms, peak extra "
              f"{r['peak_extra_gb']:.2f} GB (SDPA fp32 {r['sdpa_fp32_seconds'] * 1e3:.1f} ms)")


def parse_layers(spec):
    lo, hi = spec.split("-")[0], spec.split("-")[-1]
    return list(range(int(lo), int(hi) + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default=None, help="result file prefix (default: qwen3-<size> from the model name)")
    p.add_argument("--part", default=None, help="name of this run's part file (default: the stages)")
    p.add_argument("--stages", nargs="*", default=[], choices=["train", "ppl", "recall", "consistency", "bench"])
    p.add_argument("--merge", action="store_true")
    p.add_argument("--scorers", nargs="+", default=["order1", "omix4", "rope128", "rope128s"], choices=list(COMPONENTS))
    p.add_argument("--student-layers", type=int, nargs="+", default=RECALL_ONLY + REPLACED)
    p.add_argument("--retrain", action="store_true", help="train even if the students file exists")
    # training
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--t-train", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--max-slope", type=float, default=0.015)
    p.add_argument("--slope-lr-mult", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    # evaluation
    p.add_argument("--eval-tokens", type=int, default=9 * 32768)
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192, 16384, 32768])
    p.add_argument("--max-windows", type=int, default=0, help="cap on windows per length (0 = all); smoke only")
    p.add_argument("--no-base", action="store_true", help="skip the base-model perplexity")
    p.add_argument("--subset-lengths", type=int, nargs="*", default=[2048, 32768])
    p.add_argument("--subsets", nargs="*", default=["27", "26-27", "24-27"])
    p.add_argument("--subset-scorers", nargs="*", default=["order1", "omix4"])
    p.add_argument("--batch-tokens", type=int, default=32768,
                   help="tokens per forward in the perplexity (windows per batch = batch_tokens // N)")
    p.add_argument("--recall-lengths", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    p.add_argument("--recall-windows", type=int, nargs="*", default=[0],
                   help="windows per recall length (one value, or one per length; 0 = all)")
    p.add_argument("--recall-ks", type=int, nargs="+", default=[1, 8, 64, 256])
    p.add_argument("--q-chunk", type=int, default=512)
    p.add_argument("--bench-lengths", type=int, nargs="*", default=[8192, 16384, 32768])
    args = p.parse_args()

    tag = args.tag or "qwen3-" + args.model.split("Qwen3-")[-1].split("-")[0]
    os.makedirs(args.out, exist_ok=True)
    if args.merge:
        merge(args.out, tag)
        return
    args.part = args.part or "_".join(args.stages)
    part = Part(os.path.join(args.out, f"{tag}_part_{args.part}.json"), args.model, args)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
    model = model.cuda().eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    student_path = lambda sc: os.path.join(args.out, f"{tag}_students_{sc}.pt")  # noqa: E731
    print(f"{args.model}: stages {args.stages}, scorers {args.scorers}", flush=True)

    if "train" in args.stages:
        todo = [sc for sc in args.scorers if args.retrain or not os.path.exists(student_path(sc))]
        if todo:
            val = load_stream(tokenizer, "validation")
            print(f"training {todo} on {val.shape[0]} validation tokens, layers {args.student_layers}", flush=True)
            students, stats = train_students(model, todo, args.student_layers, val, args.steps, args.t_train,
                                             args.batch, args.lr, args.seed, args.max_slope, args.warmup,
                                             args.slope_lr_mult)
            for sc in todo:
                save_students(students[sc], student_path(sc))
                part.update("train", {sc: stats[sc]})
                kl = {k: round(v, 3) for k, v in stats[sc]["final_kl"].items()}
                print(f"{sc}: final KL per layer {json.dumps(kl)} ({stats[sc]['seconds']}s)", flush=True)
            del students
            torch.cuda.empty_cache()

    if not ({"ppl", "recall", "consistency"} & set(args.stages)) and "bench" not in args.stages:
        return
    test = load_stream(tokenizer, "test")
    print(f"test stream: {test.shape[0]} tokens, scoring the first {min(args.eval_tokens, test.shape[0])}", flush=True)
    test = test[:args.eval_tokens]
    avail = {sc: load_students(student_path(sc), "cuda") for sc in args.scorers if os.path.exists(student_path(sc))}

    if "consistency" in args.stages:
        ids = test[:2048][None].cuda()
        for sc in ("order1", "omix4"):
            if sc in avail:
                row = {"scorer": sc, "N": 2048, "layers": "22-27", **consistency(model, avail[sc], REPLACED, ids)}
                part.add("consistency", row)
                print(f"consistency {sc}: {row}", flush=True)

    if "ppl" in args.stages:
        jobs = [] if args.no_base else [("base", None)]
        jobs += [(sc, "22-27") for sc in args.scorers if sc in avail]
        base_nll = {}
        for N in args.lengths:
            for sc, ls in jobs + [(sc, s) for s in args.subsets for sc in args.subset_scorers
                                  if sc in avail and N in args.subset_lengths]:
                with contextlib.ExitStack() as stack:
                    if sc != "base":
                        stack.enter_context(replaced(model, avail[sc], parse_layers(ls)))
                    row = {"scorer": sc, "layers": ls or "", **perplexity(model, test, N, args.max_windows,
                                                                         args.batch_tokens)}
                part.add("ppl", row)
                if sc == "base":
                    base_nll[N] = row["nll"]
                gap = f", NLL gap to base {row['nll'] - base_nll[N]:+.4f}" if sc != "base" and N in base_nll else ""
                print(f"ppl N={N:6d} {sc:8s} layers {ls or '-':6s}: {row['ppl']:.3f}{gap} ({row['seconds']}s, "
                      f"peak {row.get('peak_gb')} GB)", flush=True)
                torch.cuda.empty_cache()

    if "recall" in args.stages:
        if "order1" not in avail:
            raise FileNotFoundError(student_path("order1"))
        nw = args.recall_windows * len(args.recall_lengths) if len(args.recall_windows) == 1 else args.recall_windows
        for N, w in zip(args.recall_lengths, nw):
            rows = recall_eval(model, avail["order1"], args.student_layers, test, N, w, args.recall_ks, args.q_chunk)
            for row in rows:
                part.add("recall", {"scorer": "order1", **row})
                print(f"recall N={N:6d} layer {row['layer']:2d} ({row['windows']} windows): r@8 {row['recall@8']:.4f} "
                      f"r@64 {row['recall@64']:.4f} r@256 {row['recall@256']:.4f} nonsink r@64 "
                      f"{row['nonsink_recall@64']:.4f} KL {row['kl']:.3f} ({row['seconds']}s)", flush=True)
            torch.cuda.empty_cache()

    if "bench" in args.stages:
        for N in args.bench_lengths:
            for M in (1, 4):
                row = bench_order_attention(N, model.config.num_attention_heads, model.config.head_dim, M)
                part.add("bench", row)
                print(f"bench {row}", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
