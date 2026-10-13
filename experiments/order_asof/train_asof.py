"""Order attention vs windows vs full attention on the temporal as-of task (asof_data.py), Qwen3 base models.

The base model is frozen. Its last R layers' attention can be (i) left as is (full attention, exact), (ii) restricted to
the sink + the W most recent keys (training-free window), or (iii) replaced by a student (experiments/attn_order_dim/
longctx.py's Student: 'order1' = one K = 2 order head per head through the Triton kernels, 'rope128s' = a dot-product
head with the same sink logit, run with full attention or with the window). Students are trained on the task: cross-
entropy on the answer tokens of training documents (teacher forcing), AdamW, lr warmup + cosine, TF32 matmuls.

Evaluation: teacher-forced exact match of every answer token of each query, on held-out documents of each length,
bucketed by the answer's depth (tokens between the query and the answer record); plus the full-document NLL on the
answer tokens. Long lengths (> the model's native 32k) use YaRN rope scaling (factor --yarn) for the whole run so that
every arm sees the same base model.

Usage:
  python train_asof.py --model Qwen/Qwen3-8B --data <dir> --out <dir> --arm order1 --train --eval 32768 65536 131072
  python train_asof.py ... --arm base --eval ...                 (no training)
  python train_asof.py ... --arm base --window 4096 --eval ...   (training-free window)
  python train_asof.py ... --arm rope128s --train --eval ... [--window 4096]
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "experiments", "attn_order_dim"))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import longctx  # noqa: E402
from longctx import Student, components, save_students, load_students  # noqa: E402

QB = 2048  # query block of the windowed attention
TOK = None  # tokenizer, set in main (the oracle arm parses documents)


def windowed_attention(q, k, v, scale, window, sink_logit=None):
    """Causal attention restricted to key 0 (the sink) and the ``window`` most recent keys, in query blocks.

    q, k, v: (B, H, N, d); key 0 is scored by ``sink_logit`` (B, H, N) if given, else by q . k(0).
    """
    B, H, N, d = q.shape
    out = torch.empty_like(v)
    for s in range(0, N, QB):
        e = min(N, s + QB)
        lo = max(0, s - window + 1)
        idx = torch.arange(lo, e, device=q.device)
        if lo > 0:
            idx = torch.cat([torch.zeros(1, dtype=idx.dtype, device=idx.device), idx])
        xs = torch.arange(s, e, device=q.device)[:, None]
        ok = (idx[None] <= xs) & ((xs - idx[None] < window) | (idx[None] == 0))
        sc = (q[:, :, s:e] @ k[:, :, idx].transpose(-1, -2)) * scale
        if sink_logit is not None:
            sc[..., 0] = sink_logit[:, :, s:e]
        p = torch.softmax(sc.masked_fill(~ok, float("-inf")).float(), dim=-1).to(v.dtype)
        out[:, :, s:e] = p @ v[:, :, idx]
    return out


class WindowAttention(nn.Module):
    """The layer's own attention (q/k/v/o, norms, RoPE) restricted to the sink + W most recent keys."""

    def __init__(self, attn, window):
        super().__init__()
        self.attn, self.window = attn, window

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        at = self.attn
        B, n, _ = hidden_states.shape
        shape = (B, n, -1, at.head_dim)
        q = at.q_norm(at.q_proj(hidden_states).view(shape)).transpose(1, 2)
        k = at.k_norm(at.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = at.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        k, v = repeat_kv(k, at.num_key_value_groups), repeat_kv(v, at.num_key_value_groups)
        o = windowed_attention(q, k, v, at.scaling, self.window)
        return at.o_proj(o.transpose(1, 2).reshape(B, n, -1)), None


class StudentAttention(nn.Module):
    """Attention of a layer replaced by a student (order1 through Triton; rope128s through SDPA or a window)."""

    def __init__(self, attn, student, window=0):
        super().__init__()
        self.attn, self.student, self.window = attn, student, window

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None, **kwargs):
        at = self.attn
        B, n, _ = hidden_states.shape
        v = repeat_kv(at.v_proj(hidden_states).view(B, n, -1, at.head_dim).transpose(1, 2), at.num_key_value_groups)
        pos = position_ids[0] if position_ids is not None else torch.arange(n, device=hidden_states.device)
        out = self.student(hidden_states, pos)
        if self.student.M == 0 and self.window:
            q, k = apply_rotary_pos_emb(out["q"], out["k"], *position_embeddings)
            o = windowed_attention(q.to(v.dtype), k.to(v.dtype), v, 1.0 / math.sqrt(q.shape[-1]), self.window,
                                   out["b"].to(v.dtype))
        else:
            o = longctx.student_attend(self.student, out, v.float(), position_embeddings, False, "triton")
        return at.o_proj(o.to(v.dtype).transpose(1, 2).reshape(B, n, -1)), None


ORACLE_LAM, ORACLE_MU = 5.0, 5e4  # tent slopes: a record one time unit AFTER the query time loses by MU, the answer
_ORACLE_CACHE = {}                 # scores -LAM * (its age); the later record wins only if age > MU / LAM = 1e4 units


def oracle_features(ids, spans, metas, tokenizer, key, latest=False):
    """Exact as-of order-head realizers from the TOKENS (not learned) for the document's tracked entity.

    Keys: every record of the tracked entity gets one key per value digit (digit j in band j) and a terminator key
    at its newline token (band len(value)); code(key) = band * BIG + time, BIG > every time in the document.
    Queries: the position predicting answer token k >= 1 (digit k - 1, or the newline after the last digit) thresholds
    at (k - 1) * BIG + T, so the tent score min(LAM * (code - thr), MU * (thr - code)) is maximal at the same band of
    the tracked entity's latest record with time <= T: exactly the token to emit. With ``latest`` the threshold time is
    past the end of the log (the control: right entity, right band, wrong time). Every other position queries only the
    sink. Masked keys get g = -1e12. Returns f (n, 2), g (n, 2), b (n), all fp32 CPU, cached by ``key``.
    """
    import re
    if (key, latest) in _ORACLE_CACHE:
        return _ORACLE_CACHE[(key, latest)]
    n = ids.shape[0]
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    text = tokenizer.decode(ids)
    t_max = max(int(m.group(1)) for m in re.finditer(r"(?m)^(\d+) a\d+ \d+$", text))
    t_max = max(t_max, max(m["T"] for m in metas))
    BIG = t_max + 2
    ent = metas[0]["entity"]
    g = torch.full((n, 2), -1e12)
    f = torch.zeros(n, 2)
    f[:, 0] = 1e9  # non-query positions: every key scores about -1e9, the sink (b = 0) takes all the mass
    b = torch.zeros(n)
    start, n_rec = 0, 0
    for i, t in enumerate(toks):
        if t != "Ċ":
            continue
        m = re.fullmatch(r"(\d+) (a\d+) (\d+)", tokenizer.decode(ids[start:i]).strip())
        if m and m.group(2) == ent:
            tt, val = int(m.group(1)), m.group(3)
            assert toks[i - len(val):i] == list(val), (toks[i - len(val):i], val)
            for j in range(len(val) + 1):  # digits, then the newline as the terminator
                code = j * BIG + tt
                g[i - len(val) + j, 0], g[i - len(val) + j, 1] = ORACLE_LAM * code, -ORACLE_MU * code
            n_rec += 1
        start = i + 1
    assert n_rec > 0, "no record of the tracked entity"
    for (s, e), m in zip(spans, metas):
        T = t_max + 1 if latest else m["T"]
        for k in range(1, e - s + 1):  # position s + k - 1 predicts answer token k (token 0 is the space)
            code = (k - 1) * BIG + T
            f[s + k - 1, 0], f[s + k - 1, 1] = ORACLE_LAM * code, -ORACLE_MU * code
            b[s + k - 1] = -1e9
    _ORACLE_CACHE[(key, latest)] = (f, g, b)
    return f, g, b


class OracleAttention(nn.Module):
    """The layer's own attention plus one added order head read through a zero-initialised linear adapter.

    ``mode`` 'oracle' / 'latest': the head's realizers come from [`oracle_features`][] (set in ``current`` by the
    training / evaluation loop) and its values are the layer's own value vectors (the 8 kv heads, position 0 zeroed so
    the sink contributes nothing); 'selfv': the control with the same adapter on the position's own value vectors (same
    capacity, no retrieval). Only the adapter trains.
    """

    current = None  # (f, g, b) of the document being processed

    def __init__(self, attn, n_kv, head_dim, d_model, mode="oracle"):
        super().__init__()
        self.attn, self.mode, self.n_kv, self.head_dim = attn, mode, n_kv, head_dim
        self.adapter = nn.Linear(n_kv * head_dim, d_model, bias=False)
        nn.init.zeros_(self.adapter.weight)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        at = self.attn
        B, n, _ = hidden_states.shape
        normal, _ = at(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, **kwargs)
        with torch.no_grad():
            v = at.v_proj(hidden_states).view(B, n, self.n_kv, self.head_dim).transpose(1, 2).float()
            if self.mode == "selfv":
                o = v
            else:
                from learning.order_attention_triton import order_attention_triton
                f, g, b = OracleAttention.current
                H = self.n_kv
                v = v.clone()
                v[:, :, 0] = 0.0  # key 0 is only ever the sink: its value is what non-query positions receive
                ff = f[None, None].expand(B, H, n, 2).to(v.device).contiguous()
                gg = g[None, None].expand(B, H, n, 2).to(v.device).contiguous()
                bb = b[None, None].expand(B, H, n).to(v.device).contiguous()
                o = order_attention_triton(ff, gg, bb, v)
            o = o.transpose(1, 2).reshape(B, n, -1)
        return normal + self.adapter(o).to(normal.dtype), None


@contextlib.contextmanager
def arm_context(model, layers, arm, students, window):
    orig = {l: model.model.layers[l].self_attn for l in layers}
    try:
        for l in layers:
            if arm.startswith("oracle"):
                if l == layers[0]:
                    model.model.layers[l].self_attn = students[l]  # the OracleAttention module (holds the adapter)
            elif arm == "base" and window:
                model.model.layers[l].self_attn = WindowAttention(orig[l], window)
            elif arm != "base":
                model.model.layers[l].self_attn = StudentAttention(orig[l], students[l], window)
        yield
    finally:
        for l in layers:
            model.model.layers[l].self_attn = orig[l]


@contextlib.contextmanager
def causal_without_mask(model):
    """Drop HF's explicit (N, N) mask so SDPA runs its is_causal fast path (exact for batch 1, no padding)."""
    def drop(mod, args, kwargs):
        kwargs["attention_mask"] = None
        return args, kwargs
    hooks = [layer.self_attn.register_forward_pre_hook(drop, with_kwargs=True) for layer in model.model.layers]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def answer_logits(model, ids, spans):
    """Logits at the positions that predict each answer token (teacher forcing). Returns (logits, targets)."""
    h = model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    pos = torch.cat([torch.arange(s - 1, e, device=ids.device) for s, e in spans])  # predict token t from t-1
    tgt = torch.cat([ids[s:e + 1] for s, e in spans])
    return model.lm_head(h[pos]).float(), tgt


def train(model, students, layers, data, args):
    params = [p for st in students.values() for p in st.parameters() if p.requires_grad]
    for st in students.values():
        st.train()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / args.warmup) * 0.5 * (
        1 + math.cos(math.pi * min(t, args.steps) / args.steps)))
    gen = torch.Generator().manual_seed(args.seed)
    n_docs = len(data["ids"])
    t0, acc, curve = time.time(), 0.0, []
    for step in range(args.steps):
        i = int(torch.randint(0, n_docs, (1,), generator=gen))
        ids = data["ids"][i].cuda()
        if args.arm.startswith("oracle"):
            OracleAttention.current = oracle_features(data["ids"][i], data["answer_spans"][i], data["meta"][i],
                                                      TOK, ("train", i), latest=args.arm == "oracle_latest")
        with arm_context(model, layers, args.arm, students, args.window), causal_without_mask(model):
            logits, tgt = answer_logits(model, ids, data["answer_spans"][i])
        loss = F.cross_entropy(logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        acc += loss.item()
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            n_int = (step % args.log_every) + 1
            curve.append([step + 1, acc / n_int])
            print(f"  step {step + 1}/{args.steps} ({time.time() - t0:.0f}s): answer CE {acc / n_int:.4f}", flush=True)
            acc = 0.0
    for st in students.values():
        st.eval()
    return {"steps": args.steps, "lr": args.lr, "seconds": round(time.time() - t0, 1), "curve": curve}


@torch.no_grad()
def evaluate(model, students, layers, data, args, N):
    buckets = [0, 4096, 16384, 32768, 65536, 1 << 30]
    hits = {b: [0, 0] for b in buckets[:-1]}
    nll, n_tok, t0 = 0.0, 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    for di, (ids, spans, metas) in enumerate(zip(data["ids"], data["answer_spans"], data["meta"])):
        if args.arm.startswith("oracle"):
            OracleAttention.current = oracle_features(ids, spans, metas, TOK, ("eval", N, di),
                                                      latest=args.arm == "oracle_latest")
        ids = ids.cuda()
        with arm_context(model, layers, args.arm, students, args.window), causal_without_mask(model):
            logits, tgt = answer_logits(model, ids, spans)
        nll += F.cross_entropy(logits, tgt, reduction="sum").item()
        n_tok += tgt.numel()
        pred = logits.argmax(-1)
        k = 0
        for (s, e), m in zip(spans, metas):
            L = e - s + 1
            exact = bool((pred[k:k + L] == tgt[k:k + L]).all())
            k += L
            b = max(bb for bb in buckets[:-1] if m["depth"] >= bb)
            hits[b][0] += exact
            hits[b][1] += 1
    tot = sum(v[0] for v in hits.values()), sum(v[1] for v in hits.values())
    return {"N": N, "docs": len(data["ids"]), "queries": tot[1], "exact": tot[0] / max(tot[1], 1),
            "answer_nll": nll / n_tok, "by_depth": {f">={b}": [v[0], v[1]] for b, v in hits.items()},
            "seconds": round(time.time() - t0, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arm", choices=["base", "order1", "omix4", "rope128s", "oracle", "oracle_latest", "oracle_selfv"], required=True)
    p.add_argument("--window", type=int, default=0, help="sink + W most recent keys (0 = full attention)")
    p.add_argument("--replace", type=int, default=6, help="number of final layers replaced / windowed")
    p.add_argument("--yarn", type=float, default=4.0)
    p.add_argument("--train", action="store_true")
    p.add_argument("--init-from", default=None, help="students file to start from (default: fresh)")
    p.add_argument("--distill-steps", type=int, default=300, help="warm-start distillation steps (0 = none)")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--max-slope", type=float, default=0.015)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval", type=int, nargs="*", default=[])
    p.add_argument("--max-eval-docs", type=int, default=0)
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or (args.arm + (f"_w{args.window}" if args.window else ""))
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(args.model)
    native = cfg.max_position_embeddings
    if args.yarn > 1:
        cfg.rope_scaling = {"rope_type": "yarn", "factor": args.yarn, "original_max_position_embeddings": 32768}
        cfg.max_position_embeddings = int(32768 * args.yarn)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=cfg, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    L = cfg.num_hidden_layers
    layers = list(range(L - args.replace, L))
    print(f"{args.model} ({L} layers, native {native}, yarn x{args.yarn}); arm {tag}; layers {layers}", flush=True)
    students, record = None, {"model": args.model, "arm": args.arm, "window": args.window, "layers": layers}
    spath = os.path.join(args.out, f"students_{tag}.pt")
    global TOK
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(args.model)
    if args.arm.startswith("oracle"):
        l0 = layers[0]
        mod = OracleAttention(model.model.layers[l0].self_attn, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size,
                              mode=args.arm.split("_")[-1] if "_" in args.arm else "oracle").cuda()
        if os.path.exists(spath) and not args.train:
            mod.adapter.load_state_dict(torch.load(spath))
        students = {l0: mod}
        for prm in mod.adapter.parameters():
            prm.requires_grad_(args.train)
    elif args.arm != "base":
        if args.init_from:
            students = load_students(args.init_from, "cuda")
        elif os.path.exists(spath) and not args.train:
            students = load_students(spath, "cuda")
        else:
            students = {l: Student(cfg.hidden_size, cfg.num_attention_heads, args.arm, hidden=1024,
                                   rope_dim=cfg.head_dim, max_slope=args.max_slope).cuda() for l in layers}
        for st in students.values():
            for prm in st.parameters():
                prm.requires_grad_(args.train)
    if args.train and args.arm.startswith("oracle"):
        data = torch.load(os.path.join(args.data, "train.pt"))
        print(f"training the {tag} adapter on {len(data['ids'])} docs", flush=True)
        record["train"] = train(model, students, layers, data, args)
        torch.save(students[layers[0]].adapter.state_dict(), spath)
    elif args.train:
        data = torch.load(os.path.join(args.data, "train.pt"))
        if args.distill_steps and not args.init_from:
            # warm start: distil each student from the real head's attention (KL on 2k windows of the training docs),
            # so the replaced model starts as an approximation of the base model instead of a random mixer
            stream = torch.cat(data["ids"][:200])
            print(f"distilling {tag} from the real heads: {args.distill_steps} steps at 2048 tokens", flush=True)
            st_map, stats = longctx.train_students(model, [args.arm], layers, stream, args.distill_steps, 2048, 1,
                                                   args.lr, args.seed, args.max_slope, warmup=50, hidden=1024,
                                                   log_every=50)
            students = st_map[args.arm]
            for st in students.values():
                for prm in st.parameters():
                    prm.requires_grad_(True)
            record["distill"] = {k: v for k, v in stats[args.arm].items() if k != "curve"}
            save_students(students, spath.replace(".pt", "_distilled.pt"))
        print(f"training {tag} on {len(data['ids'])} docs", flush=True)
        record["train"] = train(model, students, layers, data, args)
        save_students(students, spath)
    for N in args.eval:
        data = torch.load(os.path.join(args.data, f"eval_{N}.pt"))
        if args.max_eval_docs:
            data = {k: v[:args.max_eval_docs] for k, v in data.items()}
        r = evaluate(model, students, layers, data, args, N)
        record.setdefault("eval", []).append(r)
        print(f"eval {tag} N={N}: exact {r['exact']:.3f} ({r['queries']} queries), answer NLL {r['answer_nll']:.3f}, "
              f"by depth {r['by_depth']} ({r['seconds']}s, peak {r['peak_gb']} GB)", flush=True)
        with open(os.path.join(args.out, f"result_{tag}.json"), "w") as fh:
            json.dump(record, fh, indent=1)
        torch.cuda.empty_cache()
    with open(os.path.join(args.out, f"result_{tag}.json"), "w") as fh:
        json.dump(record, fh, indent=1)


if __name__ == "__main__":
    main()
