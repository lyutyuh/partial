"""Benchmark the linear-time arc aggregation against the O(N^2) score matrix on one GPU.

For each sentence length N (batch B), times forward+backward of the arc log-partition Z and of greedy decoding, and
records peak memory, for: the quadratic score matrix (what ModelForPartialOrder does), the pure-torch O(N log N)
Alg. 1 (sort + logcumsumexp), and the Triton kernels.

Usage (repo root, GPU node): python scripts/bench_linear_order.py [--batch 32] [--lengths 32 64 ... 4096]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning.linear_order import arc_scores_quadratic, decode, log_partition, log_partition_torch  # noqa: E402


def z_quadratic(f, g, lens):
    return arc_scores_quadratic(f, g, lens).logsumexp(-1)


def decode_quadratic(f, g, lens):
    return arc_scores_quadratic(f, g, lens).argmax(-1)


def measure(fn, f, g, lens, backward, reps):
    def step():
        if backward:
            fr, gr = f.detach().requires_grad_(), g.detach().requires_grad_()
            fn(fr, gr, lens).sum().backward()
        else:
            with torch.no_grad():
                fn(f, g, lens)

    for _ in range(3):  # warm-up, includes Triton compilation for this BLOCK
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps, (torch.cuda.max_memory_allocated() - base) / 2**20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    print(f"device: {torch.cuda.get_device_name()}, batch {args.batch}, fp32 realizers (B, N, 2)")
    print(f"{'task':<8}{'N':>6} | {'quadratic ms':>13}{'MiB':>9} | {'torch-sort ms':>14}{'MiB':>9} | "
          f"{'triton ms':>10}{'MiB':>9} | {'speedup':>8}")
    for task, backward, fns in [
        ("Z f+b", True, [z_quadratic, log_partition_torch, log_partition]),
        ("decode", False, [decode_quadratic, None, decode]),
    ]:
        for n in args.lengths:
            gen = torch.Generator(device="cuda").manual_seed(n)
            f = torch.randn(args.batch, n, 2, device="cuda", generator=gen)
            g = torch.randn(args.batch, n, 2, device="cuda", generator=gen)
            lens = torch.full((args.batch,), n, device="cuda")
            row = []
            for fn in fns:
                if fn is None:
                    row.append((float("nan"), float("nan")))
                    continue
                try:
                    row.append(measure(fn, f, g, lens, backward, args.reps))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row.append((float("inf"), float("inf")))
            (tq, mq), (ts, ms), (tt, mt) = row
            print(f"{task:<8}{n:>6} | {tq:>13.3f}{mq:>9.1f} | {ts:>14.3f}{ms:>9.1f} | {tt:>10.3f}{mt:>9.1f} | "
                  f"{tq / tt:>7.1f}x")


if __name__ == "__main__":
    main()
