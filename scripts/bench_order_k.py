"""Benchmark the general-K range-tree aggregation against the O(N^2) score matrix (and the K = 2 Triton kernel).

For each K and sentence length N (batch B): time of forward + backward of the arc log-partition Z and of greedy
decoding, plus peak memory, for the quadratic score matrix and the O(N log^(K-1) N) range tree (learning/order_k.py).
K = 2 also runs the Triton kernel (learning/linear_order.py) for reference.

Usage (repo root, GPU node): python scripts/bench_order_k.py [--batch 32] [--ks 2 3 4] [--lengths 64 ... 4096]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning.linear_order import arc_scores_quadratic, decode, log_partition  # noqa: E402
from learning.order_k import decode_k, log_partition_k  # noqa: E402


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

    for _ in range(2):
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
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--custom-backward", action="store_true", help="use the hand-written transposed backward")
    parser.add_argument("--no-decode", action="store_true")
    args = parser.parse_args()
    lpk = (lambda f, g, l: log_partition_k(f, g, l, True, autograd=False)) if args.custom_backward else log_partition_k
    print(f"device: {torch.cuda.get_device_name()}, batch {args.batch}, fp32 realizers (B, N, K), "
          f"backward = {'custom (transposed dominance)' if args.custom_backward else 'autograd'}")
    print(f"{'task':<7}{'K':>2}{'N':>6} | {'quadratic ms':>13}{'MiB':>9} | {'range-tree ms':>14}{'MiB':>9} | "
          f"{'speedup':>8} | {'triton K=2 ms':>14}")
    tasks = [("Z f+b", True, z_quadratic, lpk, log_partition)]
    if not args.no_decode:
        tasks.append(("decode", False, decode_quadratic, lambda *a: decode_k(*a)[0], lambda *a: decode(*a)[0]))
    for task, backward, fq, fk, ft in tasks:
        for K in args.ks:
            for n in args.lengths:
                gen = torch.Generator(device="cuda").manual_seed(n * 10 + K)
                f = torch.randn(args.batch, n, K, device="cuda", generator=gen)
                g = torch.randn(args.batch, n, K, device="cuda", generator=gen)
                lens = torch.full((args.batch,), n, device="cuda")
                out = []
                for fn in (fq, fk):
                    try:
                        out.append(measure(fn, f, g, lens, backward, args.reps))
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        out.append((float("inf"), float("inf")))
                tri = measure(ft, f, g, lens, backward, args.reps)[0] if K == 2 else float("nan")
                (tq, mq), (tk, mk) = out
                print(f"{task:<7}{K:>2}{n:>6} | {tq:>13.2f}{mq:>9.0f} | {tk:>14.2f}{mk:>9.0f} | {tq / tk:>7.1f}x | "
                      f"{tri:>14.2f}", flush=True)


if __name__ == "__main__":
    main()
