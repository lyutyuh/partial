"""Summarise fit.py results: how well K-order realizers reproduce each head's attention argmax.

Headline metric: non-sink top-1 accuracy (queries whose argmax key is not position 0, key 0 excluded from the
candidates; see fit.py), over heads with at least --min-nonsink such test queries.

Variants: a (layer, scorer, K, mask) can be fitted with several logit scales, step counts and learning rates (fit.py
--taus / --steps / --lr). Every variant is kept, and per layer the one with the best mean accuracy over the counted
heads is reported, selected on the held-out training sequences when every variant has them (fit.py --val-seqs),
otherwise on the test split (optimistic; the "variants" column says which). Records written before fit.py recorded
the variant used raw scores for every scorer (an unscaled dot product), 1500 steps and lr 1e-3, and are merged as that
variant. A re-run of the same variant in a later file replaces the earlier one, so a layer cut off by a wall-clock
limit and re-run elsewhere is counted once.

Reference: each scorer is also expressed relative to a learned 128-d reference, by default the rope-128 head-form
control (linear q/k + RMSNorm + RoPE, the real head's own function class) when it was fitted, else the dot-128 MLP
realizer (the order realizers' MLP features). Neither is a ceiling: with its own weights the real head reproduces its
argmax ~100% of the time, and both references are what this recipe learns from the training sequences. A ratio above 1
means a scorer beat that learned reference, not the head.

Sanity check: a wider bilinear form contains the narrower one, so the best train loss over the variants must not rise
with dot width. Layers where it does are listed (the wide fit is under-optimised and the sweep too narrow; this flagged
the unscaled dot-128 of the 2026-10-07 runs), and --strict turns them into a non-zero exit status.

Heads are typed by fixed rules on the test set: sink (>= 90% of queries -> position 0), else the best of prev / self /
induction if it explains >= 50% of queries, else "other".

Usage: python report.py results/*.jsonl [--min-nonsink 1000] [--reference auto|rope:128|dot:128] [--strict]
"""
import argparse
import collections
import json
import sys

import numpy as np

# numbers per token per side (query side also gets one gate per component for omix / ogrp); mirrors fit.py
WIDTH = {"order": lambda k: k, "dot": lambda k: k, "rope": lambda k: k, "osum": lambda m: 2 * m,
         "omix": lambda m: 2 * m, "ogrp": lambda m: 6 * m}
PER_LAYER = [("order", 2), ("osum", 4), ("osum", 16), ("omix", 16), ("ogrp", 8), ("dot", 16), ("dot", 32),
             ("dot", 64), ("dot", 128), ("rope", 128)]
REF_LABEL = {"rope": "the rope-{k} head-form control (linear q/k + RMSNorm + RoPE, learned with the same recipe)",
             "dot": "the dot-{k} MLP realizer (same MLP features as the order realizers)"}


def variant(r):
    """(score scale, steps, lr) of a fit record; older records lack the fields and used (1, 1500, 1e-3)."""
    return (float(f"{r.get('score_scale', 1.0):.6g}"), int(r.get("steps", 1500)), float(f"{r.get('lr', 1e-3):.6g}"))


def load(files):
    """{model: {(layer, scorer, K, masked, variant): record}}, files read in order, later records replacing earlier."""
    latest = collections.defaultdict(dict)
    for path in files:
        model = path.split("/")[-1].split("_p")[0]
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                var = variant(r) if r["scorer"] != "heuristics" else None
                latest[model][(r["layer"], r["scorer"], r.get("K"), r.get("masked"), var)] = r
    return latest


def mean_acc(r, ok, prefix=""):
    acc = np.array(r[prefix + "acc_nonsink"])
    return acc[ok].mean() if ok.any() else np.nan


def select(recs, min_nonsink):
    """Best variant per (layer, scorer, K, masked): {key: (record, number of variants, "val" | "test")}."""
    groups = collections.defaultdict(list)
    for (layer, scorer, k, masked, _), r in recs.items():
        if scorer != "heuristics":
            groups[(layer, scorer, k, masked)].append(r)
    out = {}
    for key, rs in groups.items():
        split = "val" if all("val_acc_nonsink" in r for r in rs) else "test"
        prefix = "val_" if split == "val" else ""
        # the heads counted in the tables (test queries >= min_nonsink); -1 so all-nan layers still pick one
        best = max(rs, key=lambda r: np.nan_to_num(
            mean_acc(r, np.array(r["n_nonsink"]) >= min_nonsink, prefix), nan=-1.0))
        out[key] = (best, len(rs), split)
    return out


def dot_width_violations(recs, rel_tol=0.1, abs_tol=0.005):
    """Layers where the lowest dot train loss over the variants rises with width beyond the tolerance (a wider dot
    contains the narrower one): [(layer, narrow K, its loss, wide K, its loss)], the worst pair per layer."""
    best = collections.defaultdict(dict)
    for (layer, scorer, k, masked, _), r in recs.items():
        if scorer == "dot" and masked:
            best[layer][k] = min(best[layer].get(k, np.inf), r["train_loss"])
    out = []
    for layer, by_k in sorted(best.items()):
        ks, worst = sorted(by_k), None
        for i, kn in enumerate(ks):
            for kw in ks[i + 1:]:
                excess = by_k[kw] - (by_k[kn] * (1 + rel_tol) + abs_tol)
                if excess > 0 and (worst is None or excess > worst[0]):
                    worst = (excess, kn, kw)
        if worst:
            out.append((layer, worst[1], by_k[worst[1]], worst[2], by_k[worst[2]]))
    return out


def head_type(h, heur):
    if heur["sink"][h] >= 0.9:
        return "sink"
    best = max(("prev", "self", "induction"), key=lambda r: heur[r][h])
    return best if heur[best][h] >= 0.5 else "other"


def reference_of(selected, spec):
    if spec == "auto":
        spec = "rope:128" if any(k[1] == "rope" and k[2] == 128 and k[3] for k in selected) else "dot:128"
    scorer, k = spec.split(":")
    return scorer, int(k)


def summarise(model, recs, args):
    """Print the tables of one model; returns its dot width violations."""
    heur = {k[0]: r["heuristics"] for k, r in recs.items() if k[1] == "heuristics"}
    selected = select(recs, args.min_nonsink)
    ref_scorer, ref_k = reference_of(selected, args.reference)
    ref = {key[0]: np.array(rec["acc_nonsink"]) for key, (rec, _, _) in selected.items()
           if key[1:] == (ref_scorer, ref_k, True)}
    types = {(l, h): head_type(h, heur[l]) for l in heur for h in range(len(heur[l]["sink"]))}
    n_types = collections.Counter(types.values())
    print(f"\n## {model}: {len(heur)} layers, {len(types)} heads; head types {dict(n_types)}")

    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    sweep = collections.defaultdict(lambda: [0, set()])  # key -> [max number of variants, selection splits]
    for (layer, scorer, k, masked), (r, n_var, split) in selected.items():
        key = (scorer, k, masked)
        sweep[key][0] = max(sweep[key][0], n_var)
        if n_var > 1:
            sweep[key][1].add(split)
        ok = np.array(r["n_nonsink"]) >= args.min_nonsink
        acc = np.array(r["acc_nonsink"])
        rel = acc / np.maximum(ref[layer], 1e-6) if layer in ref else np.full_like(acc, np.nan)
        for h in np.nonzero(ok)[0]:
            t = types[(layer, int(h))]
            for group in ("all", t):
                rows[key][group + ":acc"].append(acc[h])
                rows[key][group + ":rel"].append(rel[h])
            rows[key]["all:conf"].append(r["acc_conf"][h] if "acc_conf" in r else np.nan)

    groups = ["all"] + [t for t in ("prev", "self", "induction", "other", "sink") if n_types.get(t)]
    print(f"non-sink top-1 accuracy, mean over heads (fraction of heads >= 0.9 of the reference: "
          f"{REF_LABEL[ref_scorer].format(k=ref_k)}; a learned reference, NOT a ceiling, and a ratio above 1 beats "
          f"the reference, not the head)")
    print("| scorer | K | width | mask | heads | variants | confident-query acc | " + " | ".join(groups) + " |")
    print("|---|---|---|---|---|---|---|" + "---|" * len(groups))
    order = sorted(rows, key=lambda k: (not k[2], WIDTH[k[0]](k[1]), k[0]))
    for key in order:
        cells = []
        for g in groups:
            a, rl = rows[key].get(g + ":acc", []), np.array(rows[key].get(g + ":rel", []))
            rl = rl[~np.isnan(rl)]
            frac = f"{np.mean(rl >= 0.9):.0%}" if len(rl) else "-"
            cells.append(f"{np.mean(a):.3f} ({frac}, n={len(a)})" if a else "-")
        n_var, splits = sweep[key]
        var = f"{n_var} (best on {'/'.join(sorted(splits))})" if n_var > 1 else "1"
        print(f"| {key[0]} | {key[1]} | {WIDTH[key[0]](key[1])} | {'causal' if key[2] else 'none'} | "
              f"{len(rows[key]['all:acc'])} | {var} | "
              f"{np.nanmean(rows[key]['all:conf']):.3f} | " + " | ".join(cells) + " |")

    present = [c for c in PER_LAYER if any(k[1:3] == c for k in selected)]
    if not present:
        present = [("order", k) for k in (1, 2, 3, 4)] + [("dot", 2), ("dot", 128)]
    print("\nper-layer mean non-sink accuracy (causal, best variant): " + " ".join(f"{a}{k}" for a, k in present))
    for l in sorted(heur):
        def m(scorer, k):
            sel = selected.get((l, scorer, k, True))
            if sel is None:
                return "  -  "
            ok = np.array(sel[0]["n_nonsink"]) >= args.min_nonsink
            return f"{np.mean(np.array(sel[0]['acc_nonsink'])[ok]):.3f}" if ok.any() else "  -  "
        print(f"layer {l:2d}: " + " ".join(m(a, k) for a, k in present))

    bad = dot_width_violations(recs, args.loss_rel_tol, args.loss_abs_tol)
    if bad:
        print(f"\nWARNING: dot train loss rises with width in {len(bad)} layers (a wider dot contains the narrower "
              f"one, so the wide fit is under-optimised: widen the --taus / --steps / --lr sweep before using it as "
              f"a reference): " + "; ".join(f"L{l} dot{kn} {ln:.3f} -> dot{kw} {lw:.3f}" for l, kn, ln, kw, lw in bad))
    return bad


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--min-nonsink", type=float, default=1000)
    parser.add_argument("--reference", default="auto", help="auto (rope:128 if fitted, else dot:128) or scorer:K")
    # every width sees the same minibatches (fit.py seeds the sampler), so the tail losses compare with little noise
    parser.add_argument("--loss-rel-tol", type=float, default=0.1)
    parser.add_argument("--loss-abs-tol", type=float, default=0.005)
    parser.add_argument("--strict", action="store_true", help="exit 1 if dot train loss rises with width")
    args = parser.parse_args(argv)

    bad = {model: summarise(model, recs, args) for model, recs in sorted(load(args.files).items())}
    return 1 if args.strict and any(bad.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
