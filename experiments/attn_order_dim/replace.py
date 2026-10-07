"""Fit order scorers to attention heads by KL to the full attention distribution, then measure (a) recall@k of the
teacher's argmax under the student (indexer use) and (b) perplexity of the real model with those heads replaced.

Teacher distributions are recomputed from the stored attention inputs (extract.py) with the layer's own q/k
projections, q_norm/k_norm and RoPE, so no N x N data is stored. The student scorers are those of fit.py (order,
osum, omix, dot) applied to the attention input + sinusoidal position, plus one learned sink logit per query
(the ROOT column of the kernels), trained with KL(teacher || student) over the causal candidate set.

Replacement runs the model in eager fp32 attention and overrides the attention weights of the chosen heads with the
student's distribution (values and output projection untouched), then scores WikiText-103 test tokens.

Usage:
  python replace.py --model Qwen/Qwen3-0.6B-Base --data <extract dir> --out <jsonl> --layers 22 23 24 25 26 27 \\
      --scorers order:2 osum:2 osum:3 omix:4 dot:2 dot:8 dot:128 --perplexity --layer-sets 27 26-27 24-27 22-27
  python replace.py ... --layers 0 ... 27 --scorers order:2 osum:2 osum:3 dot:2        # recall@k only
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit import GROUP, POS_DIM, order_scores, pair_scores, sinusoid  # noqa: E402

WIDTH = {"order": lambda k: k, "dot": lambda k: k, "osum": lambda m: 2 * m, "omix": lambda m: 2 * m,
         "ogrp": lambda m: 2 * GROUP * m}
GATES = {"omix": lambda m: m, "ogrp": lambda m: m}
KS = (1, 8, 32, 64)


class Student(nn.Module):
    """Per-layer MLP -> for every head: f (width), g (width), mixture gates, and one sink logit per query."""

    def __init__(self, d_in, heads, scorer, k, hidden=1024):
        super().__init__()
        self.heads, self.scorer, self.k = heads, scorer, k
        self.width, self.gates = WIDTH[scorer](k), GATES.get(scorer, lambda _: 0)(k)
        self.net = nn.Sequential(nn.Linear(d_in + POS_DIM, hidden), nn.GELU(),
                                 nn.Linear(hidden, heads * (2 * self.width + self.gates + 1)))

    def log_probs(self, x, pos, causal):
        """log p(y | x) over keys (b, H, n, n) under the causal mask; column 0 is the learned sink logit."""
        b, n, _ = x.shape
        out = self.net(torch.cat([x, pos.expand(b, -1, -1)], dim=-1)).view(b, n, self.heads, -1).transpose(1, 2)
        w = self.width
        f, g, gate, sink = out[..., :w], out[..., w:2 * w], out[..., 2 * w:2 * w + self.gates], out[..., -1]
        mask = ~causal  # (n, n) True where disallowed
        if self.scorer in ("order", "dot", "osum"):
            if self.scorer == "order":
                s = order_scores(f, g)
            elif self.scorer == "dot":
                s = torch.einsum("bhxk,bhyk->bhxy", f, g) / math.sqrt(w)  # attention's 1/sqrt(d) scaling
            else:
                s = pair_scores(f, g, w // 2).sum(dim=-1)
            s = torch.cat([sink.unsqueeze(-1), s[..., 1:]], dim=-1)
            return torch.log_softmax(s.masked_fill(mask, float("-inf")), dim=-1)
        comp = pair_scores(f, g, w // 2)  # (b, H, n, n, M)
        if self.scorer == "ogrp":
            comp = comp.view(*comp.shape[:4], -1, GROUP).sum(dim=-1)
        comp = torch.cat([sink.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, comp.shape[-1]), comp[:, :, :, 1:]], 3)
        comp = torch.log_softmax(comp.masked_fill(mask[..., None], float("-inf")), dim=3)
        comp = comp.masked_fill(mask[..., None], -1e30)
        mix = torch.log_softmax(gate, dim=-1).unsqueeze(3)
        return torch.logsumexp(comp + mix, dim=-1).masked_fill(mask, float("-inf"))


class Teacher:
    """Recomputes a layer's attention probabilities from stored attention inputs with the model's own weights."""

    def __init__(self, model, layer):
        self.model, self.at = model, model.model.layers[layer].self_attn
        cfg = model.config
        self.H, self.Hkv, self.D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

    @torch.no_grad()
    def probs(self, x, causal):
        b, n, _ = x.shape
        pos = torch.arange(n, device=x.device)[None].expand(b, -1)
        cos, sin = self.model.model.rotary_emb(x, pos)
        q = self.at.q_norm(self.at.q_proj(x).view(b, n, self.H, self.D)).transpose(1, 2)
        k = self.at.k_norm(self.at.k_proj(x).view(b, n, self.Hkv, self.D)).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.H // self.Hkv)
        logits = (q @ k.transpose(2, 3)) * self.at.scaling
        return torch.softmax(logits.masked_fill(~causal, float("-inf")), dim=-1)


def fit_student(model, layer, train_x, pos, causal, scorer, k, steps, batch, lr, seed):
    torch.manual_seed(seed)
    teacher = Teacher(model, layer)
    student = Student(train_x.shape[-1], teacher.H, scorer, k).cuda()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tail = []
    for step in range(steps):
        idx = torch.randint(0, train_x.shape[0], (batch,), device="cuda", generator=gen)
        x = train_x[idx].float()
        p = teacher.probs(x, causal)
        lq = student.log_probs(x, pos, causal)
        kl = (p * (torch.log(p.clamp(min=1e-30)) - lq)).masked_fill(~causal, 0.0).sum(-1)  # (b, H, n)
        loss = kl[:, :, 1:].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step >= steps - 50:
            tail.append(loss.item())
    return student, teacher, sum(tail) / len(tail)


@torch.no_grad()
def evaluate_student(student, teacher, test_x, pos, causal, batch):
    H = teacher.H
    kl = torch.zeros(H, device="cuda")
    hit = {name: torch.zeros(H, device="cuda") for name in ("top1", "top1_nonsink")}
    rec = {kk: torch.zeros(H, device="cuda") for kk in KS}
    n_all = n_nonsink = 0
    for s0 in range(0, test_x.shape[0], batch):
        x = test_x[s0:s0 + batch].float()
        p = teacher.probs(x, causal)
        lq = student.log_probs(x, pos, causal)
        kl += (p * (torch.log(p.clamp(min=1e-30)) - lq)).masked_fill(~causal, 0.0).sum(-1)[:, :, 1:].sum((0, 2))
        y = p[:, :, 1:].argmax(-1)  # (b, H, n-1)
        nonsink = (y != 0).float()
        ranks = (lq[:, :, 1:] > lq[:, :, 1:].gather(-1, y.unsqueeze(-1))).sum(-1)  # keys scored above the target
        hit["top1"] += (ranks == 0).float().sum((0, 2))
        hit["top1_nonsink"] += ((ranks == 0).float() * nonsink).sum((0, 2))
        for kk in KS:
            rec[kk] += (ranks < kk).float().sum((0, 2))
        n_all += y.shape[0] * y.shape[2]
        n_nonsink += nonsink.sum((0, 2))
    out = {"kl": (kl / n_all).tolist(), "top1": (hit["top1"] / n_all).tolist(),
           "top1_nonsink": (hit["top1_nonsink"] / n_nonsink.clamp(min=1)).tolist()}
    out.update({f"recall@{kk}": (rec[kk] / n_all).tolist() for kk in KS})
    return out


class ReplacedAttention(nn.Module):
    """Eager Qwen3 attention whose weights for selected heads come from a student; values and o_proj unchanged."""

    def __init__(self, attn, student, pos, heads):
        super().__init__()
        self.attn, self.student, self.pos, self.heads = attn, student, pos, heads

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None,
                **kwargs):
        at = self.attn
        b, n, _ = hidden_states.shape
        shape = (b, n, -1, at.head_dim)
        q = at.q_norm(at.q_proj(hidden_states).view(shape)).transpose(1, 2)
        k = at.k_norm(at.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = at.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = repeat_kv(k, at.num_key_value_groups), repeat_kv(v, at.num_key_value_groups)
        causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
        logits = (q @ k.transpose(2, 3)) * at.scaling
        attn = torch.softmax(logits.masked_fill(~causal, float("-inf")).float(), dim=-1)
        if self.heads:
            lq = self.student.log_probs(hidden_states.float(), self.pos[:n], causal)
            attn[:, self.heads] = torch.exp(lq[:, self.heads])
        out = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(b, n, -1)
        return at.o_proj(out), None


@torch.no_grad()
def perplexity(model, tokens, batch):
    nll, count = 0.0, 0
    for s0 in range(0, tokens.shape[0], batch):
        ids = tokens[s0:s0 + batch].long().cuda()
        logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
        nll += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum").item()
        count += ids[:, 1:].numel()
    return math.exp(nll / count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--scorers", nargs="+", default=["order:2", "osum:2", "osum:3", "omix:4", "dot:2", "dot:8", "dot:128"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perplexity", action="store_true")
    p.add_argument("--layer-sets", nargs="*", default=["27", "26-27", "24-27", "22-27"],
                   help="layer ranges to replace jointly, e.g. 27 26-27 24-27 22-27 (all heads of those layers)")
    args = p.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager").cuda().eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    tok_test = torch.load(os.path.join(args.data, "test", "tokens.pt"))
    students = {}  # (layer, scorer, k) -> Student
    for layer in args.layers:
        train_x = torch.load(os.path.join(args.data, "train", f"layer{layer:02d}.pt"))["x"].cuda()
        test_x = torch.load(os.path.join(args.data, "test", f"layer{layer:02d}.pt"))["x"].cuda()
        n = train_x.shape[1]
        pos = sinusoid(n, POS_DIM, "cuda")
        causal = torch.ones(n, n, dtype=torch.bool, device="cuda").tril()
        for sc in args.scorers:
            scorer, k = sc.split(":")[0], int(sc.split(":")[1])
            t0 = time.time()
            student, teacher, train_kl = fit_student(model, layer, train_x, pos, causal, scorer, k, args.steps,
                                                     args.batch, args.lr, args.seed)
            metrics = evaluate_student(student, teacher, test_x, pos, causal, args.batch)
            students[(layer, scorer, k)] = student
            rec = {"model": args.model, "layer": layer, "scorer": scorer, "K": k, "width": WIDTH[scorer](k),
                   "train_kl": train_kl, "seconds": round(time.time() - t0, 1), **metrics}
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            mean = lambda v: sum(v) / len(v)
            print(f"layer {layer:2d} {scorer:5s} K={k:<3d} KL {mean(metrics['kl']):.3f} top1 {mean(metrics['top1']):.3f} "
                  f"nonsink {mean(metrics['top1_nonsink']):.3f} r@8 {mean(metrics['recall@8']):.3f} "
                  f"r@64 {mean(metrics['recall@64']):.3f} ({rec['seconds']}s)", flush=True)
        del train_x, test_x
        torch.cuda.empty_cache()

    if not args.perplexity:
        return
    n = tok_test.shape[1]
    pos = sinusoid(n, POS_DIM, "cuda")
    originals = {l: model.model.layers[l].self_attn for l in args.layers}
    base = perplexity(model, tok_test, args.batch)
    results = {"model": args.model, "base_ppl": base, "replaced": []}
    print(f"base perplexity {base:.3f}", flush=True)
    all_heads = list(range(model.config.num_attention_heads))
    for ls in args.layer_sets:
        lo, hi = (int(ls.split("-")[0]), int(ls.split("-")[-1]))
        layers = list(range(lo, hi + 1))
        for sc in args.scorers:
            scorer, k = sc.split(":")[0], int(sc.split(":")[1])
            if any((l, scorer, k) not in students for l in layers):
                continue
            for l in layers:
                model.model.layers[l].self_attn = ReplacedAttention(originals[l], students[(l, scorer, k)], pos, all_heads)
            ppl = perplexity(model, tok_test, args.batch)
            for l in layers:
                model.model.layers[l].self_attn = originals[l]
            results["replaced"].append({"layers": ls, "scorer": scorer, "K": k, "width": WIDTH[scorer](k), "ppl": ppl})
            print(f"replace layers {ls:>6s} with {scorer:5s} K={k:<3d}: ppl {ppl:.3f} (base {base:.3f})", flush=True)
    with open(args.out.replace(".jsonl", "_ppl.json"), "w") as fh:
        json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
