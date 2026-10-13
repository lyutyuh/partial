"""Mechanism test: an order head with a hashed-n-gram + time order does as-of lookup with the entity in the QUERY.

Rank code of a record y: code(y) = hash(entity) * C_T + cell(time);  rank r(y) = u * code(y).
Query (entity e, time T):       a(x) = u * (hash(e) * C_T + cell(T)).
Realizers: g1 = lam * r, g2 = g1 - r;  f1 = lam * a, f2 = f1 - a  (so score = g1 - f1 - ReLU(r - a)):
  branch 1 (r <= a): score = lam * (r - a)  -> favours the largest rank below the threshold = the most recent record of
  e with time <= T (all of e's records sit in one contiguous band of the order);
  branch 2 (r > a):  score = -(1 - lam) * (r - a) -> later records and other bands are penalised.
Non-record tokens get g = -1e9 (masked keys). V = [value, 1, 0...]; the output's first channel is the predicted value.
Runs the Triton causal order attention on real asof_data documents at 32k / 128k and reports exact-match accuracy
and the sharpness (softmax mass on the correct record), for several entity-hash sizes and time-cell counts.
"""
import re
import sys

import torch

sys.path.insert(0, "/users/tliu/repos/partial")
from learning.order_attention_triton import order_attention_triton  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DATA = "/capstor/store/cscs/swissai/a0087/tianyu/partial/order_asof/data"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def hash_entity(name, n_cells):
    h = 1469598103934665603
    for ch in name.encode():
        h = ((h ^ ch) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h % n_cells


@torch.no_grad()
def run(doc_ids, spans, metas, c_ent, c_time, u=8.0, lam=0.5):
    """Keys from the joint token sequence (split at newline tokens), so positions are exact."""
    N = doc_ids.shape[0]
    toks = tok.convert_ids_to_tokens(doc_ids.tolist())
    text = tok.decode(doc_ids)
    t_max = max(int(m.group(1)) for m in re.finditer(r"(?m)^(\d+) a\d+ \d+$", text)) + 1
    g = torch.full((N, 2), -1e9)
    V = torch.zeros(N, 8)
    V[:, 1] = 1.0
    start = 0
    for i, t in enumerate(toks):
        if "Ċ" not in t:
            continue
        line = tok.decode(doc_ids[start:i + 1]).strip()
        m = re.fullmatch(r"(\d+) (a\d+) (\d+)", line)
        if m:
            tt, ent, val = int(m.group(1)), m.group(2), int(m.group(3))
            code = hash_entity(ent, c_ent) * c_time + (tt * c_time) // t_max
            r = u * code
            g[i - 1, 0], g[i - 1, 1] = lam * r, lam * r - r  # key on the value's last digit
            V[i - 1, 0] = val
        start = i + 1
    f = torch.zeros(N, 2)
    b = torch.full((N,), -1e9)
    b[0] = 0.0
    qpos = []
    for (s, e), m in zip(spans, metas):
        code = hash_entity(m["entity"], c_ent) * c_time + (m["T"] * c_time) // t_max
        a = u * code
        f[s - 1, 0], f[s - 1, 1] = lam * a, lam * a - a
        qpos.append(s - 1)
    out = order_attention_triton(f[None, None].cuda(), g[None, None].cuda(), b[None, None].cuda(),
                                 V[None, None].cuda())[0, 0]
    pred = out[qpos, 0].cpu()
    truth = torch.tensor([m["answer"] for m in metas], dtype=torch.float32)
    exact = ((pred - truth).abs() < 0.5).float()
    return exact, (pred - truth).abs()


@torch.no_grad()
def run_tent(doc_ids, spans, metas, c_time, lam, mu):
    """Relevance given (keys = the queried entity's records only); g1 = lam*cell, g2 = -mu*cell; f1 = lam*cell(T),
    f2 = -mu*cell(T): score = min(lam*(c_y - c_T), mu*(c_T - c_y)), a tent with slope lam below T and mu above."""
    N = doc_ids.shape[0]
    toks = tok.convert_ids_to_tokens(doc_ids.tolist())
    text = tok.decode(doc_ids)
    t_max = max(int(m.group(1)) for m in re.finditer(r"(?m)^(\d+) a\d+ \d+$", text)) + 1
    ent = metas[0]["entity"]
    g = torch.full((N, 2), -1e9)
    V = torch.zeros(N, 8)
    V[:, 1] = 1.0
    start = 0
    for i, t in enumerate(toks):
        if "Ċ" not in t:
            continue
        m = re.fullmatch(r"(\d+) (a\d+) (\d+)", tok.decode(doc_ids[start:i + 1]).strip())
        if m and m.group(2) == ent:
            c = (int(m.group(1)) * c_time) // t_max
            g[i - 1, 0], g[i - 1, 1] = lam * c, -mu * c
            V[i - 1, 0] = int(m.group(3))
        start = i + 1
    f = torch.zeros(N, 2)
    b = torch.full((N,), -1e9)
    b[0] = 0.0
    qpos = []
    for (s_, e_), m in zip(spans, metas):
        c = (m["T"] * c_time) // t_max
        f[s_ - 1, 0], f[s_ - 1, 1] = lam * c, -mu * c
        qpos.append(s_ - 1)
    out = order_attention_triton(f[None, None].cuda(), g[None, None].cuda(), b[None, None].cuda(), V[None, None].cuda())[0, 0]
    pred = out[qpos, 0].cpu()
    truth = torch.tensor([m["answer"] for m in metas], dtype=torch.float32)
    return ((pred - truth).abs() < 0.5).float(), (pred - truth).abs()


def main():
    for N in (32768, 131072):
        d = torch.load(f"{DATA}/eval_{N}.pt")
        for c_time, lam, mu in ((4096, 5.0, 500.0), (65536, 5.0, 500.0), (65536, 20.0, 2000.0)):
            ex, err = [], []
            for i in range(16):
                e, a = run_tent(d["ids"][i], d["answer_spans"][i], d["meta"][i], c_time, lam, mu)
                ex.append(e); err.append(a)
            ex, err = torch.cat(ex), torch.cat(err)
            print(f"TENT (relevance given) N={N:>6} time cells {c_time:>5} lam={lam:g} mu={mu:g}: exact {ex.mean():.3f} "
                  f"over {ex.numel()} queries, median |err| {err.median():.4f}, max {err.max():.2f}", flush=True)
    return

    for N in (32768, 131072):
        d = torch.load(f"{DATA}/eval_{N}.pt")
        # lam, u from the expected gap (in time cells) between consecutive records of the queried entity:
        # need lam*u*gap >= 5 (previous record loses) and (1-lam)*u >= lam*u*gap + 5 (a record one cell after T loses)
        for c_ent, c_time, lam, u in ((256, 4096, 1 / 68, 12.0), (1024, 1024, 0.06, 12.0), (256, 4096, 0.5, 8.0)):
            note = " (lam=0.5: the first run's regime)" if lam == 0.5 else f" (lam={lam:.3f}, u={u:g})"
            if c_ent * c_time * u > 2 ** 24 * 1.01:
                note += " [code range exceeds fp32 precision]"
            ex, err = [], []
            for i in range(16):
                e, a = run(d["ids"][i], d["answer_spans"][i], d["meta"][i], c_ent, c_time, u, lam)
                ex.append(e)
                err.append(a)
            ex, err = torch.cat(ex), torch.cat(err)
            print(f"N={N:>6} entity cells {c_ent:>5} time cells {c_time:>5}: exact {ex.mean():.3f} over {ex.numel()} "
                  f"queries, median |err| {err.median():.3f}, max {err.max():.1f}{note}", flush=True)


if __name__ == "__main__":
    main()
