"""Benchmark causal order attention (learning/order_attention.py) against dense attention, for prefill and decoding.

Prefill: forward + backward of one layer's attention output (B sequences, H heads, head dim d) for the position-tree
order attention vs the dense reference (explicit N x N logits, like eager attention) and vs PyTorch SDPA with a
standard dot-product head (what a production layer costs). Decoding: per-token latency of the logarithmic-method
``OrderCache`` vs a KV cache (q . K^T over the cache, softmax, @ V) at several context lengths.

Usage (repo root, GPU node): python scripts/bench_order_attention.py [--lengths 512 ... 16384] [--decode-lengths ...]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning.order_attention import OrderCache, dense_order_attention, order_attention  # noqa: E402


def timed(fn, reps):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000, (torch.cuda.max_memory_allocated() - base) / 2**20


def bench_prefill(args):
    B, H, d = args.batch, args.heads, args.dim
    print(f"\nprefill, forward + backward, B={B} H={H} d={d} fp32 (SDPA in bf16 as a production layer would run)")
    print(f"{'N':>6} | {'dense order ms':>14}{'MiB':>8} | {'tree order ms':>14}{'MiB':>8} | {'speedup':>8} | {'SDPA dot ms':>12}")
    for n in args.lengths:
        gen = torch.Generator(device="cuda").manual_seed(n)
        f = torch.randn(B, H, n, 2, device="cuda", generator=gen)
        g = torch.randn(B, H, n, 2, device="cuda", generator=gen)
        b = torch.randn(B, H, n, device="cuda", generator=gen)
        V = torch.randn(B, H, n, d, device="cuda", generator=gen)
        q = torch.randn(B, H, n, d, device="cuda", generator=gen, dtype=torch.bfloat16)
        k = torch.randn(B, H, n, d, device="cuda", generator=gen, dtype=torch.bfloat16)

        def run(fn):
            ts = [t.detach().requires_grad_() for t in (f, g, b, V)]
            fn(*ts).sum().backward()

        def run_sdpa():
            ts = [t.detach().requires_grad_() for t in (q, k, V.bfloat16())]
            F.scaled_dot_product_attention(*ts, is_causal=True).float().sum().backward()

        try:
            td, md = timed(lambda: run(dense_order_attention), args.reps)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            td, md = float("inf"), float("inf")
        tt, mt = timed(lambda: run(lambda *a: order_attention(*a, True, args.chunk)), args.reps)
        ts, _ = timed(run_sdpa, args.reps)
        print(f"{n:>6} | {td:>14.1f}{md:>8.0f} | {tt:>14.1f}{mt:>8.0f} | {td / tt:>7.1f}x | {ts:>12.2f}", flush=True)


def bench_decode(args):
    H, d = args.heads, args.dim
    print(f"\ndecoding, per-token latency, H={H} d={d}, one sequence, fp32")
    print(f"{'N':>7} | {'KV cache ms':>12} | {'order cache ms':>15} | {'avg over 256 steps':>18} | {'cache MiB':>10}")
    for n in args.decode_lengths:
        gen = torch.Generator(device="cuda").manual_seed(n)
        g = torch.randn(H, n, 2, device="cuda", generator=gen)
        V = torch.randn(H, n, d, device="cuda", generator=gen)
        K = torch.randn(H, n, d, device="cuda", generator=gen)
        cache = OrderCache(chunk=args.chunk)
        cache.prefill(g, V)
        torch.cuda.synchronize()
        mem = sum(t.numel() * 4 for blk in cache.blocks for t in blk.values()) / 2**20
        f1 = torch.randn(H, 2, device="cuda", generator=gen)
        b1 = torch.randn(H, device="cuda", generator=gen)
        q1 = torch.randn(H, 1, d, device="cuda", generator=gen)

        def kv_step():
            att = torch.softmax(q1 @ K.transpose(1, 2) / d**0.5, dim=-1)
            return att @ V

        def order_step(i):
            return cache.step(f1, g[:, i % n], b1, V[:, i % n])

        t_kv, _ = timed(kv_step, args.reps)
        # order cache: time 256 consecutive steps (insertions + merges + queries), report the mean
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(256):
            order_step(i)
        torch.cuda.synchronize()
        t_avg = (time.perf_counter() - t0) / 256 * 1000
        t_one, _ = timed(lambda: cache.query(f1, b1), args.reps)  # query alone, at the current size
        print(f"{n:>7} | {t_kv:>12.3f} | {t_one:>15.3f} | {t_avg:>18.3f} | {mem:>10.0f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--decode-lengths", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--no-prefill", action="store_true")
    p.add_argument("--no-decode", action="store_true")
    args = p.parse_args()
    print(f"device: {torch.cuda.get_device_name()}")
    if not args.no_prefill:
        bench_prefill(args)
    if not args.no_decode:
        bench_decode(args)


if __name__ == "__main__":
    main()
