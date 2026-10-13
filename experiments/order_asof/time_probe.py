"""How precisely can a probe on Qwen3-8B hidden states read the realizers an as-of order head needs?

The oracle head of train_asof.py ranks the tracked entity's records by their TIME and thresholds each query at its
time T; a learned head must produce both from hidden states. This measures, at several layers of the frozen model on
the 16k training documents (fit docs, then held-out docs):

1. time regression at the key position of every record (ridge, an MLP, and a digit-wise linear probe that classifies
   each decimal digit of the time and composes the number) and at the query position of every answer; errors in time
   units and relative to the gap between consecutive records of the tracked entity (what the tent must resolve);
2. the key gate: a logistic probe for "this record belongs to the watched entity";
3. the as-of retrieval accuracy a tent head would reach with the probed values: per query, the tracked record with
   the largest predicted time <= the predicted T is the answer or not (with the true gate and with the probed gate).

Usage: python time_probe.py --out <dir> [--layers 12 18 24 30] [--fit-docs 64] [--eval-docs 16] [--pos newline|timedigit]
"""
import argparse
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/users/tliu/repos/partial/experiments/order_asof")
from train_asof import causal_without_mask  # noqa: E402

DATA = "/capstor/store/cscs/swissai/a0087/tianyu/partial/order_asof/data"
N_DIGITS = 6


def parse_records(ids, tokenizer):
    """Every record line of a document: (newline position, last time-digit position, time, entity)."""
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    recs, start = [], 0
    for i, t in enumerate(toks):
        if t != "Ċ":
            continue
        line = tokenizer.decode(ids[start:i]).strip()
        m = re.fullmatch(r"(\d+) (a\d+) (\d+)", line)
        if m:
            tpos = start + len(m.group(1)) - 1  # the time digits open the line, one token each
            assert toks[tpos] == m.group(1)[-1] and toks[tpos + 1] == "Ġa", (toks[start:i], line)
            recs.append((i, tpos, int(m.group(1)), m.group(2)))
        start = i + 1
    return recs


def query_positions(ids, spans, metas, tokenizer, pos_kind):
    """Per query: the probe position (first answer position, or the last digit of T) and T."""
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    out = []
    for (s, e), m in zip(spans, metas):
        if pos_kind == "newline":
            out.append((s, m["T"]))
        else:
            p = s - 1  # 'Ġ='
            while toks[p] != "Ġa":
                p -= 1
            assert toks[p - 1] == str(m["T"])[-1], (toks[p - 8:s], m["T"])
            out.append((p - 1, m["T"]))
    return out


@torch.no_grad()
def collect(model, tokenizer, data, docs, layers, pos_kind):
    """Hidden states entering the attention of each layer (after input_layernorm) at key and query positions."""
    feats = {l: {"key": [], "query": []} for l in layers}
    meta = {"key": [], "query": []}  # key: (doc, time, tracked, record index); query: (doc, T, answer record index)
    cap = {}
    hooks = [model.model.layers[l].self_attn.register_forward_pre_hook(
        lambda m, a, k, l=l: cap.__setitem__(l, (a[0] if a else k["hidden_states"])[0]), with_kwargs=True)
        for l in layers]
    try:
        for di in docs:
            ids, spans, metas = data["ids"][di], data["answer_spans"][di], data["meta"][di]
            recs = parse_records(ids, tokenizer)
            tracked = metas[0]["entity"]
            qs = query_positions(ids, spans, metas, tokenizer, pos_kind)
            with causal_without_mask(model):
                model.model(input_ids=ids[None].cuda(), use_cache=False)
            kpos = torch.tensor([r[0] if pos_kind == "newline" else r[1] for r in recs])
            qpos = torch.tensor([q[0] for q in qs])
            for l in layers:
                h = cap[l]
                feats[l]["key"].append(h[kpos.cuda()].to(torch.float16).cpu())
                feats[l]["query"].append(h[qpos.cuda()].to(torch.float16).cpu())
            for j, r in enumerate(recs):
                meta["key"].append((di, r[2], int(r[3] == tracked), j))
            # the answer record index among this document's records (record_index counts log lines, = our order)
            for (p, T), m in zip(qs, metas):
                meta["query"].append((di, T, m["record_index"]))
            cap.clear()
    finally:
        for h in hooks:
            h.remove()
    for l in layers:
        feats[l] = {k: torch.cat(v) for k, v in feats[l].items()}
    meta = {k: torch.tensor(v) for k, v in meta.items()}
    return feats, meta


def ridge_fit(X, y, lams, Xv, yv):
    """Closed-form ridge on standardised features; the penalty chosen on the validation set. Returns a predictor."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs, Xvs = (X - mu) / sd, (Xv - mu) / sd
    ym = y.mean()
    G = Xs.T @ Xs
    best = None
    for lam in lams:
        w = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=G.device), Xs.T @ (y - ym))
        err = ((Xvs @ w + ym) - yv).abs().mean().item()
        if best is None or err < best[0]:
            best = (err, lam, w)
    _, lam, w = best
    return (lambda Z: ((Z - mu) / sd) @ w + ym), lam


def mlp_fit(X, y, Xv, yv, steps=3000, hidden=512, seed=0):
    """A 2-layer MLP regressor (Adam, full batch), early-stopped on the validation set."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ys = y.std() + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1)).to(X.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best = (float("inf"), None)
    for step in range(steps):
        pred = net((X - mu) / sd).squeeze(-1) * ys + y.mean()
        loss = F.smooth_l1_loss(pred / ys, y / ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                err = ((net((Xv - mu) / sd).squeeze(-1) * ys + y.mean()) - yv).abs().mean().item()
            if err < best[0]:
                best = (err, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    net.eval()
    return lambda Z: net((Z - mu) / sd).squeeze(-1).detach() * ys + y.mean()


def digit_fit(X, y, Xv, yv, steps=3000, seed=0):
    """Linear digit-wise probe: N_DIGITS softmax heads over 0-9 (leading zeros for short times), composed to a number."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    digits = torch.stack([(y.long() // 10 ** k) % 10 for k in range(N_DIGITS)], 1)
    W = torch.nn.Linear(X.shape[1], N_DIGITS * 10).to(X.device)
    opt = torch.optim.Adam(W.parameters(), lr=1e-3, weight_decay=1e-4)

    def predict(Z):
        logits = W((Z - mu) / sd).view(Z.shape[0], N_DIGITS, 10)
        d = logits.argmax(-1)
        return sum(d[:, k] * 10 ** k for k in range(N_DIGITS)).float(), logits

    best = (float("inf"), None)
    for step in range(steps):
        logits = W((X - mu) / sd).view(X.shape[0], N_DIGITS, 10)
        loss = F.cross_entropy(logits.reshape(-1, 10), digits.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                err = (predict(Xv)[0] - yv).abs().mean().item()
            if err < best[0]:
                best = (err, {k: v.clone() for k, v in W.state_dict().items()})
    W.load_state_dict(best[1])
    with torch.no_grad():
        dv = torch.stack([(yv.long() // 10 ** k) % 10 for k in range(N_DIGITS)], 1)
        per_digit = (predict(Xv)[1].argmax(-1) == dv).float().mean(0).tolist()
    return (lambda Z: predict(Z)[0].detach()), per_digit


def logistic_fit(X, y, Xv, yv, steps=2000, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    w = torch.nn.Linear(X.shape[1], 1).to(X.device)
    opt = torch.optim.Adam(w.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_w = ((1 - y).sum() / y.sum().clamp(min=1)).to(X.device)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(w((X - mu) / sd).squeeze(-1), y, pos_weight=pos_w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = w((Xv - mu) / sd).squeeze(-1)
        pred = (p > 0).float()
        tp = (pred * yv).sum().item()
        prec = tp / max(pred.sum().item(), 1)
        rec = tp / max(yv.sum().item(), 1)
        acc = (pred == yv).float().mean().item()
    return (lambda Z: (w((Z - mu) / sd).squeeze(-1) > 0).float().detach()), {"acc": acc, "precision": prec, "recall": rec}


def retrieval(meta_k, meta_q, t_key, t_query, gate):
    """As-of retrieval with predicted times: per query, the gated record with the largest predicted time <= predicted T."""
    hits = tot = 0
    for qi in range(meta_q.shape[0]):
        di, T, ans = meta_q[qi].tolist()
        sel = (meta_k[:, 0] == di) & (gate > 0.5)
        idx = sel.nonzero().flatten()
        if idx.numel() == 0:
            tot += 1
            continue
        tk = t_key[idx]
        below = tk <= t_query[qi]
        j = int(tk.masked_fill(~below, float("-inf")).argmax()) if below.any() else int(tk.argmin())
        hits += int(meta_k[idx[j], 3] == ans)
        tot += 1
    return hits / max(tot, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layers", type=int, nargs="+", default=[12, 18, 24, 30])
    p.add_argument("--fit-docs", type=int, default=64)
    p.add_argument("--eval-docs", type=int, default=16)
    p.add_argument("--pos", choices=["newline", "timedigit"], default="newline")
    p.add_argument("--yarn", type=float, default=4.0)
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"probe_{args.pos}_fit{args.fit_docs}"
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    if args.yarn > 1:
        cfg.rope_scaling = {"rope_type": "yarn", "factor": args.yarn, "original_max_position_embeddings": 32768}
        cfg.max_position_embeddings = int(32768 * args.yarn)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=cfg, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    data = torch.load(os.path.join(DATA, "train.pt"))
    t0 = time.time()
    fit_docs = list(range(args.fit_docs))
    eval_docs = list(range(args.fit_docs, args.fit_docs + args.eval_docs))
    feats, meta = collect(model, tok, data, fit_docs + eval_docs, args.layers, args.pos)
    print(f"collected {meta['key'].shape[0]} records ({int(meta['key'][:, 2].sum())} tracked) and {meta['query'].shape[0]} "
          f"queries from {len(fit_docs) + len(eval_docs)} docs in {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    kfit = torch.isin(meta["key"][:, 0], torch.tensor(fit_docs))
    qfit = torch.isin(meta["query"][:, 0], torch.tensor(fit_docs))
    mk, mq = meta["key"][~kfit], meta["query"][~qfit]
    # the resolution the tent needs: gaps between consecutive tracked records in the eval docs
    gaps = []
    for di in eval_docs:
        ts = mk[(mk[:, 0] == di) & (mk[:, 2] == 1), 1].tolist()
        gaps += [b - a for a, b in zip(ts, ts[1:])]
    gaps = torch.tensor(gaps, dtype=torch.float)
    results = {"pos": args.pos, "fit_docs": args.fit_docs, "eval_docs": args.eval_docs,
               "tracked_gap_median": gaps.median().item(), "tracked_gap_p10": gaps.quantile(0.1).item(),
               "t_max_median": float(torch.tensor([mk[mk[:, 0] == di, 1].max().item() for di in eval_docs]).median()),
               "layers": {}}
    print(f"tracked-record gap: median {results['tracked_gap_median']:.0f}, p10 {results['tracked_gap_p10']:.0f} time units; "
          f"t_max median {results['t_max_median']:.0f}", flush=True)
    yk, yq = meta["key"][:, 1].float().cuda(), meta["query"][:, 1].float().cuda()
    gk = meta["key"][:, 2].float().cuda()
    kfit_c, qfit_c = kfit.cuda(), qfit.cuda()
    for l in args.layers:
        t1 = time.time()
        Xk, Xq = feats[l]["key"].float().cuda(), feats[l]["query"].float().cuda()
        Xk_f, Xk_v, yk_f, yk_v = Xk[kfit_c], Xk[~kfit_c], yk[kfit_c], yk[~kfit_c]
        Xq_f, Xq_v, yq_f, yq_v = Xq[qfit_c], Xq[~qfit_c], yq[qfit_c], yq[~qfit_c]
        # one shared regressor for keys and queries (a real head applies the same map at both roles); fit on both
        Xa_f, ya_f = torch.cat([Xk_f, Xq_f]), torch.cat([yk_f, yq_f])
        Xa_v, ya_v = torch.cat([Xk_v, Xq_v]), torch.cat([yk_v, yq_v])
        res = {}
        probes = {}
        ridge, lam = ridge_fit(Xa_f, ya_f, [1e1, 1e2, 1e3, 1e4, 1e5], Xa_v, ya_v)
        probes["ridge"] = ridge
        res["ridge_lambda"] = lam
        probes["mlp"] = mlp_fit(Xa_f, ya_f, Xa_v, ya_v)
        digit, per_digit = digit_fit(Xa_f, ya_f, Xa_v, ya_v)
        probes["digits"] = digit
        res["digit_acc_per_place"] = [round(a, 4) for a in per_digit]
        gate, gstats = logistic_fit(Xk_f, gk[kfit_c], Xk_v, gk[~kfit_c])
        res["gate"] = gstats
        g_true = gk[~kfit_c].cpu()
        g_hat = gate(Xk_v).cpu()
        for name, fn in probes.items():
            tk, tq = fn(Xk_v).cpu(), fn(Xq_v).cpu()
            ek, eq = (tk - yk_v.cpu()).abs(), (tq - yq_v.cpu()).abs()
            tracked = g_true > 0.5
            res[name] = {"key_mae": ek.mean().item(), "key_mae_tracked": ek[tracked].mean().item(),
                         "key_err_p90": ek.quantile(0.9).item(), "query_mae": eq.mean().item(),
                         "key_err_over_gap_median": (ek[tracked].median() / gaps.median()).item(),
                         "exact_key": (ek < 0.5).float().mean().item(), "exact_query": (eq < 0.5).float().mean().item(),
                         "retrieval_true_gate": retrieval(mk, mq, tk, tq, g_true),
                         "retrieval_probed_gate": retrieval(mk, mq, tk, tq, g_hat)}
        res["retrieval_true_times_true_gate"] = retrieval(mk, mq, yk_v.cpu(), yq_v.cpu(), g_true)
        res["retrieval_true_times_probed_gate"] = retrieval(mk, mq, yk_v.cpu(), yq_v.cpu(), g_hat)
        results["layers"][l] = res
        print(f"layer {l} ({time.time() - t1:.0f}s): gate acc {gstats['acc']:.4f} P {gstats['precision']:.3f} R {gstats['recall']:.3f}; "
              f"digit acc per place {res['digit_acc_per_place']}; "
              + "; ".join(f"{n}: key MAE {r['key_mae']:.0f} (tracked {r['key_mae_tracked']:.0f}, p90 {r['key_err_p90']:.0f}), "
                          f"query MAE {r['query_mae']:.0f}, exact key/query {r['exact_key']:.3f}/{r['exact_query']:.3f}, "
                          f"retrieval {r['retrieval_true_gate']:.3f} (probed gate {r['retrieval_probed_gate']:.3f})"
                          for n, r in res.items() if isinstance(r, dict) and "key_mae" in r)
              + f"; true times: retrieval {res['retrieval_true_times_true_gate']:.3f} (probed gate "
              f"{res['retrieval_true_times_probed_gate']:.3f})", flush=True)
        with open(os.path.join(args.out, f"{tag}.json"), "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
