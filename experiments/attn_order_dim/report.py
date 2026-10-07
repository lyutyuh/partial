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

# numbers per token per side (query side also gets one gate per component for omix / ogrp); mirrors fit.py
WIDTH = {"order": lambda k: k, "dot": lambda k: k, "osum": lambda m: 2 * m, "omix": lambda m: 2 * m,
         "ogrp": lambda m: 6 * m}
PER_LAYER = [("order", 2), ("osum", 4), ("osum", 16), ("omix", 16), ("ogrp", 8), ("dot", 16), ("dot", 32),
             ("dot", 64), ("dot", 128)]


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

    # files are read in the given order; a (layer, scorer, K, mask) re-run in a later file replaces the earlier one,
    # so a layer cut off by a wall-clock limit and re-run elsewhere is counted once
    latest = collections.defaultdict(dict)
    for path in args.files:
        model = path.split("/")[-1].split("_p")[0]
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                latest[model][(r["layer"], r["scorer"], r.get("K"), r.get("masked"))] = r
    by_model = {model: list(recs.values()) for model, recs in latest.items()}

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
        print(f"| scorer | K | width | mask | heads | confident-query acc | " + " | ".join(groups) + " |")
        print("|---|---|---|---|---|---|" + "---|" * len(groups))
        order = sorted(rows, key=lambda k: (not k[2], WIDTH[k[0]](k[1]), k[0]))
        for key in order:
            cells = []
            for g in groups:
                a, rl = rows[key].get(g + ":acc", []), rows[key].get(g + ":rel", [])
                cells.append(f"{np.mean(a):.3f} ({np.mean(np.array(rl) >= 0.9):.0%}, n={len(a)})" if a else "-")
            print(f"| {key[0]} | {key[1]} | {WIDTH[key[0]](key[1])} | {'causal' if key[2] else 'none'} | "
                  f"{len(rows[key]['all:acc'])} | "
                  f"{np.mean(rows[key]['all:conf']):.3f} | " + " | ".join(cells) + " |")

        present = [c for c in PER_LAYER if any(r["scorer"] == c[0] and r["K"] == c[1] for r in fits)]
        if not present:
            present = [("order", k) for k in (1, 2, 3, 4)] + [("dot", 2), ("dot", 128)]
        print("\nper-layer mean non-sink accuracy (causal): " + " ".join(f"{a}{k}" for a, k in present))
        for l in sorted(heur):
            def m(scorer, k, masked=True):
                rr = [r for r in fits if r["layer"] == l and r["scorer"] == scorer and r["K"] == k and
                      r["masked"] == masked]
                if not rr:
                    return "  -  "
                ok = np.array(rr[0]["n_nonsink"]) >= args.min_nonsink
                return f"{np.mean(np.array(rr[0]['acc_nonsink'])[ok]):.3f}" if ok.any() else "  -  "
            print(f"layer {l:2d}: " + " ".join(m(a, k) for a, k in present))


if __name__ == "__main__":
    main()
