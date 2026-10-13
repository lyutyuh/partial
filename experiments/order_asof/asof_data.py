"""Temporal "as-of" lookup over long event logs: the synthetic task for the order-attention 8B test.

A document is: a header naming ONE tracked entity, then a long log of timestamped records for many entities (the
tracked one with probability p), then Q queries "as of t=T, <entity>=" whose answer is the value of the tracked
entity's most recent record with time <= T. Later records of the same entity (time > T) exist in the context, so the
answer is NOT the latest record: a sliding window fails whenever the answer record is older than the window, while a
K = 2 order head represents the task directly (threshold on time, salience = relevance + recency).

Format (all ASCII, deterministic from the seed):
    watch a1187\n
    317 a42 511\n                        (records "<time> <entity> <value>", times strictly increasing)
    ...
    asof 123400 a1187 = 42\n             (queries; the answer is the integer after '=')

Usage:
    python asof_data.py --out <dir> --tokenizer Qwen/Qwen3-8B --n-train 2000 --train-tokens 16384 \
        --n-eval 64 --eval-tokens 32768 65536 131072
Writes <out>/{train,eval_<N>}.pt: dict(ids (list of 1-D int tensors), answer_spans (list of [(start, end)] per doc:
token positions of the answer digits), meta (list of per-query dicts: T, answer, depth in tokens from the query to the
answer record, entity)).
"""
import argparse
import os
import random

import torch


def make_doc(rng, n_records, p_tracked, n_queries, n_entities=2000):
    tracked = f"a{rng.randrange(n_entities)}"
    others = [f"a{i}" for i in range(n_entities)]
    t = 0
    lines = [f"watch {tracked}\n"]
    tracked_hist = []  # (time, value, record index)
    for i in range(n_records):
        t += rng.randrange(1, 40)
        if rng.random() < p_tracked:
            ent, val = tracked, rng.randrange(0, 1000)
            tracked_hist.append((t, val, i))
        else:
            ent, val = rng.choice(others), rng.randrange(0, 1000)
            if ent == tracked:
                ent = others[(others.index(ent) + 1) % n_entities]
        lines.append(f"{t} {ent} {val}\n")
    queries = []
    for _ in range(n_queries):
        # pick the answer record uniformly over the tracked history (so depths are spread over the whole log), then a
        # query time between that record and the next tracked record
        j = rng.randrange(len(tracked_hist))
        t_ans, v_ans, idx = tracked_hist[j]
        t_next = tracked_hist[j + 1][0] if j + 1 < len(tracked_hist) else t + 1
        T = rng.randrange(t_ans, t_next)
        queries.append({"T": T, "answer": v_ans, "record_index": idx, "entity": tracked})
    return lines, queries


def render(lines, queries, tokenizer):
    """Tokenise a document; returns ids (1-D LongTensor), answer spans [(start, end)] and record token offsets."""
    text = "".join(lines)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    # token offset of each record line (for depth measurement)
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(tokenizer(line, add_special_tokens=False)["input_ids"])
    assert pos == len(ids) or abs(pos - len(ids)) <= len(lines) // 100 + 2  # tokeniser may merge across newlines
    spans, metas = [], []
    for q in queries:
        prompt = f"asof {q['T']} {q['entity']} ="
        ans = f" {q['answer']}\n"
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tokenizer(ans, add_special_tokens=False)["input_ids"]
        start = len(ids) + len(p_ids)
        ids = ids + p_ids + a_ids
        spans.append((start, start + len(a_ids) - 1))  # exclude the trailing newline from the loss
        metas.append({**q, "depth": start - offsets[q["record_index"] + 1]})  # +1: line 0 is the header
    return torch.tensor(ids, dtype=torch.long), spans, metas


def build(tokenizer, n_docs, target_tokens, n_queries, p_tracked, seed, tokens_per_record=11):
    rng = random.Random(seed)
    docs = {"ids": [], "answer_spans": [], "meta": []}
    n_records = int(max(50, (target_tokens - 40 * n_queries) // tokens_per_record))
    # calibrate on one rendered document (digit counts grow along the log), aiming 2% below the target
    lines, queries = make_doc(random.Random(seed + 7), n_records, p_tracked, n_queries)
    n_records = int(n_records * 0.98 * target_tokens / render(lines, queries, tokenizer)[0].shape[0])
    for _ in range(n_docs):
        lines, queries = make_doc(rng, n_records, p_tracked, n_queries)
        ids, spans, metas = render(lines, queries, tokenizer)
        docs["ids"].append(ids)
        docs["answer_spans"].append(spans)
        docs["meta"].append(metas)
    return docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--train-tokens", type=int, default=16384)
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--eval-tokens", type=int, nargs="+", default=[32768, 65536, 131072])
    p.add_argument("--queries", type=int, default=8)
    p.add_argument("--p-tracked", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)
    # calibrate tokens per record on a sample
    rng = random.Random(123)
    lines, _ = make_doc(rng, 200, args.p_tracked, 1)
    tpr = len(tok("".join(lines[1:]), add_special_tokens=False)["input_ids"]) / 200
    print(f"tokens per record: {tpr:.2f}")
    train = build(tok, args.n_train, args.train_tokens, args.queries, args.p_tracked, args.seed, tpr)
    lens = [x.shape[0] for x in train["ids"]]
    print(f"train: {len(lens)} docs, tokens min/mean/max {min(lens)}/{sum(lens) / len(lens):.0f}/{max(lens)}, "
          f"depth median {sorted(m['depth'] for ms in train['meta'] for m in ms)[len(lens) * args.queries // 2]}")
    torch.save(train, os.path.join(args.out, "train.pt"))
    for N in args.eval_tokens:
        ev = build(tok, args.n_eval, N, args.queries, args.p_tracked, args.seed + N, tpr)
        lens = [x.shape[0] for x in ev["ids"]]
        depths = sorted(m["depth"] for ms in ev["meta"] for m in ms)
        print(f"eval {N}: {len(lens)} docs, tokens min/mean/max {min(lens)}/{sum(lens) / len(lens):.0f}/{max(lens)}, "
              f"depth quartiles {depths[len(depths) // 4]} / {depths[len(depths) // 2]} / {depths[3 * len(depths) // 4]}")
        torch.save(ev, os.path.join(args.out, f"eval_{N}.pt"))


if __name__ == "__main__":
    main()
