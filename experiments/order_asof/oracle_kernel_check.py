"""Retrieval exactness of the oracle as-of head through the Triton kernel on the real documents (GPU, seconds).

Values are [token id, 1, 0, ...], so the head's first output channel at a query position is the token id it retrieves;
it must equal the next token of the document (the digit or newline to emit). Prints the exact rate, the max error and
the kernel's sink leakage at 32k and 128k.
Usage: python oracle_kernel_check.py [--docs 4]
"""
import argparse
import sys

import torch

sys.path.insert(0, "/users/tliu/repos/partial/experiments/order_asof")
sys.path.insert(0, "/users/tliu/repos/partial/experiments/attn_order_dim")
sys.path.insert(0, "/users/tliu/repos/partial")
from learning.order_attention_triton import order_attention_triton  # noqa: E402
import train_asof  # noqa: E402

DATA = "/capstor/store/cscs/swissai/a0087/tianyu/partial/order_asof/data"


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    for N in (32768, 131072):
        d = torch.load(f"{DATA}/eval_{N}.pt")
        for latest in (False, True):
            hit = tot = 0
            worst, leak = 0.0, 0.0
            for di in range(args.docs):
                ids, spans, metas = d["ids"][di], d["answer_spans"][di], d["meta"][di]
                f, g, b = train_asof.oracle_features(ids, spans, metas, tok, ("check", N, di), latest=latest)
                n = ids.shape[0]
                V = torch.zeros(n, 8)
                V[:, 0] = ids.float()
                V[:, 1] = 1.0
                V[0] = 0.0
                H = args.heads
                out = order_attention_triton(f[None, None].expand(1, H, n, 2).contiguous().cuda(),
                                             g[None, None].expand(1, H, n, 2).contiguous().cuda(),
                                             b[None, None].expand(1, H, n).contiguous().cuda(),
                                             V[None, None].expand(1, H, n, 8).contiguous().cuda())[0, 0].cpu()
                q = (b < 0).nonzero().flatten()
                err = (out[q, 0] - ids[q + 1].float()).abs()
                hit += int((err < 0.5).sum())
                tot += q.numel()
                worst = max(worst, err.max().item())
                nonq = (b >= 0).nonzero().flatten()
                leak = max(leak, out[nonq].abs().max().item())
            print(f"N={N:>6} {'LATEST' if latest else 'ORACLE'}: retrieved token == next token {hit}/{tot} "
                  f"({hit / tot:.4f}), max |err| {worst:.3f}, non-query output max {leak:.2e}", flush=True)


if __name__ == "__main__":
    main()
