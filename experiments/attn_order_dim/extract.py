"""Extract per-head attention argmax targets and attention inputs from a causal LM on WikiText-103.

For every layer l and head h, the target of query position x is the key y*(x) = argmax_y A_lh[x, y] (causal, y <= x)
together with its weight max_y A_lh[x, y]. The features are the layer's attention input (the output of
``layers[l].input_layernorm``), i.e. exactly what the head's query/key projections read.

Writes to <out>/<split>/: tokens.pt (S, N) int32; layer{l:02d}.pt = {"x": (S, N, d) bf16, "tgt": (H, S, N) int16,
"pmax": (H, S, N) fp16}.

Usage: python extract.py --model Qwen/Qwen3-0.6B-Base --out <dir> [--seq-len 512] [--max-train 544] [--max-test 128]
"""
import argparse
import glob
import os

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WIKI = "Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/{split}-00000-of-00001.parquet"


def load_chunks(tokenizer, split, seq_len, max_chunks):
    path = glob.glob(os.path.join(os.environ["HF_HUB_CACHE"], "datasets--" + WIKI.format(split=split)))[0]
    text = "".join(pq.read_table(path).column("text").to_pylist())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n = min(len(ids) // seq_len, max_chunks)
    return torch.tensor(ids[: n * seq_len], dtype=torch.int32).view(n, seq_len)


@torch.no_grad()
def extract(model, tokens, out_dir, batch):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(tokens, os.path.join(out_dir, "tokens.pt"))
    layers = model.model.layers
    S, N = tokens.shape
    H = model.config.num_attention_heads
    feats = [torch.empty(S, N, model.config.hidden_size, dtype=torch.bfloat16) for _ in layers]
    tgts = [torch.empty(H, S, N, dtype=torch.int16) for _ in layers]
    pmaxs = [torch.empty(H, S, N, dtype=torch.float16) for _ in layers]
    cur = {}
    hooks = [layer.input_layernorm.register_forward_hook(
        lambda mod, inp, out, l=l: cur.__setitem__(l, out.detach())) for l, layer in enumerate(layers)]
    for s in range(0, S, batch):
        ids = tokens[s:s + batch].long().cuda()
        attns = model(input_ids=ids, output_attentions=True, use_cache=False).attentions  # L x (b, H, N, N)
        for l, a in enumerate(attns):
            p, y = a.float().max(dim=-1)
            tgts[l][:, s:s + batch] = y.transpose(0, 1).to(torch.int16).cpu()
            pmaxs[l][:, s:s + batch] = p.transpose(0, 1).half().cpu()
            feats[l][s:s + batch] = cur[l].to(torch.bfloat16).cpu()
        print(f"  {out_dir}: {min(s + batch, S)}/{S} sequences", flush=True)
    for h in hooks:
        h.remove()
    for l in range(len(layers)):
        torch.save({"x": feats[l], "tgt": tgts[l], "pmax": pmaxs[l]}, os.path.join(out_dir, f"layer{l:02d}.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-train", type=int, default=544)
    parser.add_argument("--max-test", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # fp32 + eager so the argmax is taken over the exact softmax probabilities
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 attn_implementation="eager").cuda().eval()
    for split, max_chunks, name in [("validation", args.max_train, "train"), ("test", args.max_test, "test")]:
        tokens = load_chunks(tokenizer, split, args.seq_len, max_chunks)
        print(f"{args.model} {name}: {tuple(tokens.shape)} from wikitext-103 {split}", flush=True)
        extract(model, tokens, os.path.join(args.out, name), args.batch)


if __name__ == "__main__":
    main()
