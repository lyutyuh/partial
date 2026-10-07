"""Summarise fit.py results: how well K-order realizers reproduce each head's attention argmax.

Headline metric: non-sink top-1 accuracy (queries whose argmax key is not position 0, key 0 excluded from the
candidates; see fit.py), over heads with at least
--min-nonsink such test queries. Each scorer is also expressed relative to the dot-128 ceiling (same features,
the real head width). Heads are typed by fixed rules on the test set: sink (>= 90% of queries -> position 0), else the
best of prev / self / induction if it explains >= 50% of queries, else "other".

Usage: python report.py results/*.jsonl [--min-nonsink 1000]
"""
import argparse
import collections
import json

import numpy as np


def head_type(h, heur):
    if heur["sink"][h] >= 0.9:
        return "sink"
    best = max(("prev", "self", "induction"), key=lambda r: heur[r][h])
    return best if heur[best][h] >= 0.5 else "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--min-nonsink", type=float, default=1000)
    args = parser.parse_args()

    by_model = collections.defaultdict(list)
    for path in args.files:
        model = path.split("/")[-1].split("_p")[0]
        with open(path) as fh:
            by_model[model] += [json.loads(line) for line in fh]

    for model, recs in sorted(by_model.items()):
        heur = {r["layer"]: r["heuristics"] for r in recs if r["scorer"] == "heuristics"}
        fits = [r for r in recs if r["scorer"] != "heuristics"]
        ceiling = {r["layer"]: np.array(r["acc_nonsink"]) for r in fits if r["scorer"] == "dot" and r["K"] == 128}
        types = {(l, h): head_type(h, heur[l]) for l in heur for h in range(len(heur[l]["sink"]))}
        n_types = collections.Counter(types.values())
        print(f"\n## {model}: {len(heur)} layers, {len(types)} heads; head types {dict(n_types)}")

        rows = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in fits:
            key = (r["scorer"], r["K"], r["masked"])
            ok = np.array(r["n_nonsink"]) >= args.min_nonsink
            acc = np.array(r["acc_nonsink"])
            rel = acc / np.maximum(ceiling.get(r["layer"], np.ones_like(acc)), 1e-6)
            for h in np.nonzero(ok)[0]:
                t = types[(r["layer"], int(h))]
                for group in ("all", t):
                    rows[key][group + ":acc"].append(acc[h])
                    rows[key][group + ":rel"].append(rel[h])
                rows[key]["all:conf"].append(r["acc_conf"][h])

        groups = ["all"] + [t for t in ("prev", "self", "induction", "other", "sink") if n_types.get(t)]
        print("non-sink top-1 accuracy, mean over heads (fraction of heads >= 0.9 of the dot-128 ceiling)")
        print(f"| scorer | K | mask | heads | confident-query acc | " + " | ".join(groups) + " |")
        print("|---|---|---|---|---|" + "---|" * len(groups))
        order = sorted(rows, key=lambda k: (k[0] != "order", not k[2], k[1]))
        for key in order:
            cells = []
            for g in groups:
                a, rl = rows[key].get(g + ":acc", []), rows[key].get(g + ":rel", [])
                cells.append(f"{np.mean(a):.3f} ({np.mean(np.array(rl) >= 0.9):.0%}, n={len(a)})" if a else "-")
            print(f"| {key[0]} | {key[1]} | {'causal' if key[2] else 'none'} | {len(rows[key]['all:acc'])} | "
                  f"{np.mean(rows[key]['all:conf']):.3f} | " + " | ".join(cells) + " |")

        print("\nper-layer mean non-sink accuracy: order K=1..4 (causal) | dot K=2 | dot-128")
        for l in sorted(heur):
            def m(scorer, k, masked=True):
                rr = [r for r in fits if r["layer"] == l and r["scorer"] == scorer and r["K"] == k and
                      r["masked"] == masked]
                if not rr:
                    return "  -  "
                ok = np.array(rr[0]["n_nonsink"]) >= args.min_nonsink
                return f"{np.mean(np.array(rr[0]['acc_nonsink'])[ok]):.3f}" if ok.any() else "  -  "
            print(f"layer {l:2d}: " + " ".join(m("order", k) for k in (1, 2, 3, 4)) +
                  f" | {m('dot', 2)} | {m('dot', 128)}")


if __name__ == "__main__":
    main()
