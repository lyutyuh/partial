"""Decode-step latency and memory of the Fenwick cache (learning/fenwick_cache.py) vs ``OrderCache`` and a KV cache.

One sequence, H order heads (one K = 2 order per query head), head dim d, fp32 states. For each context length N:
prefill N tokens, then time consecutive decode steps (insert the new key, attend) of ``OrderFenwickCache`` at several
cell counts B (eager and CUDA-graph), of ``OrderCache`` (CUDA-graph; its steps include the rare binary-counter merges,
so the p99 is reported next to the median), and of a bf16 GQA KV cache serving the H query heads over ``--kv-heads``
KV heads (write k / v into slot N, SDPA over N + 1 keys; CUDA-graph). The prefill (build) time and the cache bytes are
reported too: the Fenwick cache is the only one whose bytes do not grow with N.

Usage (GPU): python scripts/bench_fenwick_cache.py [--lengths 4096 16384 65536 131072] [--cells 256 4096 65536]
    [--heads 16] [--kv-heads 8] [--dim 128] [--steps 200]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning.fenwick_cache import OrderFenwickCache, quantile_edges  # noqa: E402
from learning.order_attention import OrderCache  # noqa: E402


def timed_steps(fn, steps, warmup=5):
    """Median and p99 latency (ms) of ``fn(i)`` over ``steps`` calls after ``warmup`` calls."""
    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize()
    times = []
    for i in range(warmup, warmup + steps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(i)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = torch.tensor(times)
    return t.median().item(), t.quantile(0.99).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 65536, 131072])
    p.add_argument("--cells", type=int, nargs="+", default=[256, 4096, 65536])
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    H, kvh, d, S = args.heads, args.kv_heads, args.dim, args.steps
    gen = torch.Generator(device="cuda").manual_seed(0)
    print(f"{H} order heads (one K=2 order per query head), d={d}, fp32 states; KV baseline: {H} query heads over "
          f"{kvh} KV heads, bf16; {S} timed steps per cell", flush=True)
    print(f"{'N':>7} {'cache':>14} {'B':>6} | {'build ms':>8} | {'step med':>8} {'step p99':>8} {'query':>7} | "
          f"{'MiB':>8}", flush=True)
    for N in args.lengths:
        g_pre = torch.randn(H, N, 2, device="cuda", generator=gen)
        v_pre = torch.randn(H, N, d, device="cuda", generator=gen)
        T = S + 5
        f_dec = torch.randn(T, H, 2, device="cuda", generator=gen)
        g_dec = torch.randn(T, H, 2, device="cuda", generator=gen)
        b_dec = torch.randn(T, H, device="cuda", generator=gen)
        v_dec = torch.randn(T, H, d, device="cuda", generator=gen)
        ranks = g_pre[..., 0] - g_pre[..., 1]
        for B in args.cells:
            edges = quantile_edges(ranks, B)
            for name, kw in (("fenwick eager", {}), ("fenwick graph", {"cuda_graph": True}),
                             ("fenwick triton", {"impl": "triton"})):
                cache = OrderFenwickCache(edges, **kw)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                cache.prefill(g_pre, v_pre)
                torch.cuda.synchronize()
                build = (time.perf_counter() - t0) * 1000
                med, p99 = timed_steps(lambda i: cache.step(f_dec[i], g_dec[i], b_dec[i], v_dec[i]), S)
                qmed, _ = timed_steps(lambda i: cache.query(f_dec[i], b_dec[i]), S)
                print(f"{N:>7} {name:>14} {B:>6} | {build:>8.1f} | {med:>8.3f} {p99:>8.3f} {qmed:>7.3f} | "
                      f"{cache.nbytes() / 2**20:>8.1f}", flush=True)
                del cache
        cache = OrderCache(cuda_graph=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cache.prefill(g_pre, v_pre)
        torch.cuda.synchronize()
        build = (time.perf_counter() - t0) * 1000
        med, p99 = timed_steps(lambda i: cache.step(f_dec[i], g_dec[i], b_dec[i], v_dec[i]), S)
        qmed, _ = timed_steps(lambda i: cache.query(f_dec[i], b_dec[i]), S)
        print(f"{N:>7} {'order graph':>14} {'-':>6} | {build:>8.1f} | {med:>8.3f} {p99:>8.3f} {qmed:>7.3f} | "
              f"{cache.nbytes() / 2**20:>8.1f}", flush=True)
        del cache
        # KV GQA bf16: write the new k / v into slot N, attend over N + 1 keys (CUDA graph)
        K = torch.randn(1, kvh, N + 1, d, device="cuda", generator=gen, dtype=torch.bfloat16)
        V = torch.randn(1, kvh, N + 1, d, device="cuda", generator=gen, dtype=torch.bfloat16)
        q_in = torch.zeros(1, H, 1, d, device="cuda", dtype=torch.bfloat16)
        k_in = torch.zeros(1, kvh, d, device="cuda", dtype=torch.bfloat16)
        v_in = torch.zeros(1, kvh, d, device="cuda", dtype=torch.bfloat16)
        q_dec = torch.randn(T, 1, H, 1, d, device="cuda", generator=gen, dtype=torch.bfloat16)
        kv_new = torch.randn(T, 2, 1, kvh, d, device="cuda", generator=gen, dtype=torch.bfloat16)

        def kv_step():
            K[:, :, N] = k_in
            V[:, :, N] = v_in
            return F.scaled_dot_product_attention(q_in, K, V, enable_gqa=True)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            kv_step()
        torch.cuda.current_stream().wait_stream(stream)
        kv_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(kv_graph):
            kv_out = kv_step()

        def kv_fn(i):
            torch._foreach_copy_([q_in, k_in, v_in], [q_dec[i], kv_new[i, 0], kv_new[i, 1]])
            kv_graph.replay()
            return kv_out.clone()

        med, p99 = timed_steps(kv_fn, S)
        print(f"{N:>7} {'kv gqa graph':>14} {'-':>6} | {'-':>8} | {med:>8.3f} {p99:>8.3f} {'-':>7} | "
              f"{2 * kvh * N * d * 2 / 2**20:>8.1f}", flush=True)
        del K, V, kv_graph
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
