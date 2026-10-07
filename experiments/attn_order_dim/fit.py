"""Fit K-order realizers (and matched dot-product scorers) to each attention head's argmax graph.

For layer l, an MLP maps every token's attention input (+ sinusoidal position) to 2K numbers per head:
f(x) (query side) and g(y) (key side). Scorers:
    order:  s(x, y) = -max_k (f_k(x) - g_k(y))     (Eq. 2 of Liu et al., EMNLP 2023; order dimension K)
    dot:    s(x, y) = <f(x), g(y)>                  (rank-K bilinear, same 2K numbers per token)
Combinations of M (or L) K = 2 order heads, to approximate one wide head:
    osum:   s(x, y) = sum_m s_m(x, y)   ("AND", product of experts; exactly one structured K = 2^M realizer)
    omix:   p(y | x) = sum_m alpha_m(x) p_m(y | x)   ("OR", query-gated mixture of L heads' distributions)
    ogrp:   omix over L groups, each group an osum of 3 heads
trained with cross-entropy towards the head's argmax key y*(x), with the causal mask (y <= x) or without it
("unmasked": the realizer must itself rank future keys below the target).

The attention sink (y* = 0) is trivially realizable (one key with a huge score) and is the argmax of most queries, so
it would dominate the fit. Training and evaluation therefore use only non-sink queries (y* != 0) and drop key 0 from
the candidates: the question is the order dimension of the remaining, non-trivial argmax relation. Held-out top-1
accuracy per head is reported on these queries and on the confident subset (max attention >= 0.5), along with simple
head-type heuristics (sink, previous token, self, induction) computed on all queries.

Usage: python fit.py --data <extract out dir> --layers 0 1 2 --out results.jsonl [--steps 1500]
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch import nn

CONFIGS = [("order", k, True) for k in (1, 2, 3, 4)] + [("dot", k, True) for k in (1, 2, 3, 4)] + \
          [("dot", 128, True), ("order", 2, False), ("order", 3, False)]
POS_DIM = 64
GROUP = 3  # heads per group in ogrp
# per-token width of f (= of g) and number of query gates, for each scorer and its size parameter
WIDTH = {"order": lambda k: k, "dot": lambda k: k, "osum": lambda m: 2 * m, "omix": lambda m: 2 * m,
         "ogrp": lambda m: 2 * GROUP * m}
GATES = {"omix": lambda m: m, "ogrp": lambda m: m}


def sinusoid(n, dim, device):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    return torch.cat([torch.sin(pos * freq), torch.cos(pos * freq)], dim=-1)


class Realizer(nn.Module):
    def __init__(self, d_in, heads, width, gates=0, hidden=1024):
        super().__init__()
        self.heads, self.width, self.gates = heads, width, gates
        self.net = nn.Sequential(nn.Linear(d_in + POS_DIM, hidden), nn.GELU(),
                                 nn.Linear(hidden, heads * (2 * width + gates)))

    def forward(self, x, pos):
        b, n, _ = x.shape
        out = self.net(torch.cat([x, pos.expand(b, -1, -1)], dim=-1)).view(b, n, self.heads, -1).transpose(1, 2)
        w = self.width
        return out[..., :w], out[..., w:2 * w], out[..., 2 * w:]  # f, g: (b, H, n, w); gate: (b, H, n, gates)


def order_scores(f, g):
    """-max_k (f_k(x) - g_k(y)) over the last dim: (b, H, n, K) x2 -> (b, H, n, n)."""
    return -(f.unsqueeze(3) - g.unsqueeze(2)).amax(dim=-1)


def pair_scores(f, g, n_heads):
    """Per-component K = 2 order scores: (b, H, n, 2M) x2 -> (b, H, n, n, M)."""
    b, H, n, _ = f.shape
    fm = f.view(b, H, n, n_heads, 2).permute(0, 1, 3, 2, 4)
    gm = g.view(b, H, n, n_heads, 2).permute(0, 1, 3, 2, 4)
    return -(fm.unsqueeze(4) - gm.unsqueeze(3)).amax(dim=-1).permute(0, 1, 3, 4, 2)


def allowed_mask(n, masked, device):
    allowed = torch.ones(n, n, dtype=torch.bool, device=device)
    if masked:
        allowed = allowed.tril()
    allowed[:, 0] = False  # the sink key is excluded; sink queries are not scored
    return allowed[1:]  # queries x >= 1


def log_probs(f, g, gate, scorer, masked):
    """log p(y | x) over candidate keys for queries x >= 1: (b, H, n - 1, n), -inf outside the candidates."""
    n = f.shape[2]
    ok = allowed_mask(n, masked, f.device)
    if scorer in ("order", "dot", "osum"):
        if scorer == "order":
            s = order_scores(f, g)
        elif scorer == "dot":
            s = torch.einsum("bhxk,bhyk->bhxy", f, g)
        else:
            s = pair_scores(f, g, f.shape[-1] // 2).sum(dim=-1)
        return torch.log_softmax(s[:, :, 1:].masked_fill(~ok, float("-inf")), dim=-1)
    comp = pair_scores(f, g, f.shape[-1] // 2)  # (b, H, n, n, heads)
    if scorer == "ogrp":
        comp = comp.view(*comp.shape[:4], -1, GROUP).sum(dim=-1)  # (b, H, n, n, L groups)
    comp = torch.log_softmax(comp[:, :, 1:].masked_fill(~ok[..., None], float("-inf")), dim=3)
    # keys outside the candidates are -inf in every component; keep them finite through the logsumexp (whose gradient
    # is NaN on an all -inf slice) and restore -inf afterwards
    comp = comp.masked_fill(~ok[..., None], -1e30)
    mix = torch.log_softmax(gate[:, :, 1:], dim=-1).unsqueeze(3)  # (b, H, n - 1, 1, L)
    return torch.logsumexp(comp + mix, dim=-1).masked_fill(~ok, float("-inf"))


def induction_targets(tok):
    """y = p + 1 for the most recent p < x with tok[p] == tok[x]; -1 if the current token has not occurred."""
    n = tok.shape[-1]
    eq = (tok[:, :, None] == tok[:, None, :]) & torch.ones(n, n, dtype=torch.bool, device=tok.device).tril(-1)
    idx = torch.arange(n, device=tok.device)
    last = torch.where(eq, idx, -1).amax(dim=-1)
    return torch.where(last >= 0, last + 1, -1)


def heuristics(tgt, tok):
    """Per-head accuracy of fixed attention rules on queries x >= 1. tgt: (H, S, N)."""
    n = tgt.shape[-1]
    x = torch.arange(n, device=tgt.device)
    rules = {"sink": torch.zeros_like(x), "prev": x - 1, "self": x}
    out = {name: (tgt[..., 1:] == y[1:]).float().mean(dim=(1, 2)).tolist() for name, y in rules.items()}
    ind = induction_targets(tok)  # (S, N)
    out["induction"] = (tgt[..., 1:] == ind[None, :, 1:]).float().mean(dim=(1, 2)).tolist()
    return out


def fit_one(train, test, pos, scorer, k, masked, steps, batch, lr, seed):
    torch.manual_seed(seed)
    H = train["tgt"].shape[0]
    model = Realizer(train["x"].shape[-1], H, WIDTH[scorer](k), GATES.get(scorer, lambda _: 0)(k)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    S = train["x"].shape[0]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    loss_tail = []
    for step in range(steps):
        idx = torch.randint(0, S, (batch,), device="cuda", generator=gen)
        lp = log_probs(*model(train["x"][idx].float(), pos), scorer, masked)
        y = train["tgt"][:, idx, 1:].long().transpose(0, 1)  # (b, H, n - 1)
        y = y.masked_fill(y == 0, -100)  # sink queries are ignored
        loss = F.nll_loss(lp.reshape(-1, lp.shape[-1]), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step >= steps - 50:
            loss_tail.append(loss.item())

    model.eval()
    correct = {name: torch.zeros(H, device="cuda") for name in ("nonsink", "conf")}
    count = {name: torch.zeros(H, device="cuda") for name in ("nonsink", "conf")}
    with torch.no_grad():
        for s0 in range(0, test["x"].shape[0], batch):
            pred = log_probs(*model(test["x"][s0:s0 + batch].float(), pos), scorer, masked).argmax(dim=-1)
            y = test["tgt"][:, s0:s0 + batch, 1:].long().transpose(0, 1)
            p = test["pmax"][:, s0:s0 + batch, 1:].float().transpose(0, 1)
            hit = (pred == y).float()
            nonsink = (y != 0).float()
            for name, m in (("nonsink", nonsink), ("conf", nonsink * (p >= 0.5).float())):
                correct[name] += (hit * m).sum(dim=(0, 2))
                count[name] += m.sum(dim=(0, 2))
    acc = {f"acc_{name}": (correct[name] / count[name].clamp(min=1)).tolist() for name in correct}
    acc.update({f"n_{name}": count[name].tolist() for name in ("conf", "nonsink")})
    return acc, sum(loss_tail) / max(len(loss_tail), 1)


def load(path):
    d = torch.load(path)
    return {"x": d["x"].cuda(), "tgt": d["tgt"].cuda(), "pmax": d["pmax"].cuda()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--configs", nargs="*", default=None,
                        help="subset, e.g. order:2:masked dot:128:masked osum:4:masked omix:8:masked ogrp:4:masked")
    args = parser.parse_args()

    configs = CONFIGS
    if args.configs:
        configs = [(c.split(":")[0], int(c.split(":")[1]), c.split(":")[2] == "masked") for c in args.configs]
    tok_test = torch.load(os.path.join(args.data, "test", "tokens.pt")).cuda()
    for layer in args.layers:
        train = load(os.path.join(args.data, "train", f"layer{layer:02d}.pt"))
        test = load(os.path.join(args.data, "test", f"layer{layer:02d}.pt"))
        pos = sinusoid(train["x"].shape[1], POS_DIM, "cuda")
        base = {"layer": layer, "pmax_mean": test["pmax"][..., 1:].float().mean(dim=(1, 2)).tolist(),
                "heuristics": heuristics(test["tgt"].long(), tok_test.long())}
        with open(args.out, "a") as fh:
            fh.write(json.dumps({**base, "scorer": "heuristics"}) + "\n")
        for scorer, k, masked in configs:
            t0 = time.time()
            acc, loss = fit_one(train, test, pos, scorer, k, masked, args.steps, args.batch, args.lr, args.seed)
            rec = {"layer": layer, "scorer": scorer, "K": k, "masked": masked, "train_loss": loss,
                   "seconds": round(time.time() - t0, 1), **acc}
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            mean = sum(acc["acc_nonsink"]) / len(acc["acc_nonsink"])
            print(f"layer {layer:2d} {scorer:5s} K={k:<3d} {'masked' if masked else 'unmasked':8s} "
                  f"non-sink acc {mean:.3f} loss {loss:.3f} ({rec['seconds']}s)", flush=True)
        del train, test
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
