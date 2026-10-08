"""Benchmark causal order attention (learning/order_attention.py) against dense attention, for prefill and decoding.

Prefill: forward + backward of one layer's attention output (B sequences, H heads, head dim d) for the position-tree
order attention vs the dense reference (explicit N x N logits, like eager attention) and vs PyTorch SDPA with a
standard dot-product head (what a production layer costs). Decoding: ``OrderCache`` (static block + logarithmic tail,
eager and CUDA-graph) and the previous ``OrderCacheSimple`` vs a KV cache at several context lengths: query latency,
per-step latency distribution over consecutive decode steps, and memory (see ``bench_decode``).

What the decode comparison is: by default ONE K = 2 order component per query head (16 heads, fp32) against KV caches
for the same 16 query heads, both MHA (16 KV heads) and GQA (``--kv-heads`` 8, as in Qwen3-0.6B / 1.7B), eager and
replayed from CUDA graphs. A single K = 2 head costs ~1-2 ppl on Qwen3; the quality-matched omix-4 replacement needs
``--components 4`` (64 order heads), which multiplies the order cache's memory and work by 4.

Usage (repo root, GPU node): python scripts/bench_order_attention.py [--lengths 512 ... 16384] [--decode-lengths ...]
    [--kv-heads 8] [--components 1]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning.order_attention import (  # noqa: E402
    OrderCache, OrderCacheSimple, dense_order_attention, order_attention,
)


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


def _events(fn, reps):
    """Mean ms per call over ``reps`` back-to-back calls (CUDA events; launches overlap GPU work, as inside a model)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def _graphed(fn, *static):
    """``fn(*static)`` captured in a CUDA graph. The returned callable copies its arguments into ``static`` (one
    launch), replays and clones the output: the same per-call work as a graph-mode ``OrderCache.query`` / ``step``."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*static)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(*static)

    def run(*args):
        torch._foreach_copy_(list(static), list(args))
        graph.replay()
        return out.clone()
    return run


def _sync_steps(fn, steps, peak_step=None, held=None):
    """Per-call ms of fn(0) .. fn(steps - 1), synchronised around every call (perf_counter).

    With ``peak_step``: also the peak allocated MiB during that call, counted from the allocated memory before it minus
    ``held()`` bytes (the caller's own buffers, which the call may free), so the peak includes those buffers.
    """
    ts, peak = [], None
    for i in range(steps):
        torch.cuda.synchronize()
        if i == peak_step:
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated() - held()
        t = time.perf_counter()
        fn(i)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
        if i == peak_step:
            peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    return torch.tensor(ts, dtype=torch.float64) * 1000, peak


def _steps(cache, f, g, b, V, n, steps, pool, peak_step=None):
    """Per-step ms of ``steps`` consecutive decode steps after an n-token prefill, synchronised around every step, and
    the cache's peak MiB during step ``peak_step``.

    The decoded tokens cycle through a pool of ``pool`` fresh random tokens stored after the prefill in g / V.
    """
    def step(i):
        k = i % pool
        cache.step(f[:, k], g[:, n + k], b[:, k], V[:, n + k])
    return _sync_steps(step, steps, peak_step, getattr(cache, "nbytes", None))


def _reserved_after_free():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.memory_reserved()


def _stats(ts):
    return f"{ts.mean():>7.3f}{ts.quantile(0.5):>7.3f}{ts.quantile(0.99):>7.3f}{ts.max():>8.2f}"


def bench_decode(args):
    """Decoding with the order cache vs a KV cache, one sequence, head dim d.

    Configuration: the order cache (fp32) holds H x M heads, one K = 2 component per query head and component
    (``--components`` M; the quality-matched omix-4 replacement of the Qwen3 layers needs M = 4). The KV baselines serve
    the H query heads over H KV heads (MHA) or over ``--kv-heads`` KV heads (GQA; Qwen3-0.6B / 1.7B have 16 query heads
    over 8 KV heads, so GQA is the cache the order cache would replace). The order cache cannot share values across the
    query heads of a GQA group (each head sorts its keys by its own rank), so its memory ratio to the GQA cache is the
    relevant one.

    (a) query latency alone: ``OrderCache.query`` eager and CUDA-graph, ``OrderCacheSimple.query`` (block list);
    (b) per-step latency over ``--steps`` consecutive decode steps, synchronised around every step. Order cache (insert
        + merge + query): eager right after the prefill; graph after ``--graph-warmup`` steps that capture the step
        graphs of merge levels 0..8; simple (its step 1 is the cascade). KV GQA bf16 step: write the new k / v into
        slot N of a preallocated cache and run SDPA over the N + 1 keys, replayed from a CUDA graph with the same input
        copy and output clone as the graph-mode order cache step;
    (c) KV-cache query baselines for one query token over N cached keys, each eager and replayed from a CUDA graph (so
        that graph-mode order queries are compared with graph-mode KV queries): fp32 manual (q K^T, softmax, @ V; MHA),
        bf16 SDPA MHA and bf16 SDPA GQA;
    (d) memory: order cache buffers (``nbytes``: static block + preallocated tail) plus the memory pool that its CUDA
        graphs share (reserved-memory growth over the captures of the query graph and the step graphs of levels 0..8)
        vs the KV caches, and the transient peaks of a prefill and of a fold step (max allocated, buffers included);
    (e) one full tail cycle in graph mode: the C + 1 steps from the prefill to the first fold, which include the graph
        captures at the start (a fold reallocates the buffer, so every cycle recaptures), every merge level, the eager
        deep merges and the fold itself: their mean is the amortised per-token cost.
    """
    H, M, kvh, d = args.heads, args.components, args.kv_heads, args.dim
    S, W, P = args.steps, args.graph_warmup, args.pool
    Ho, gqa = H * M, args.kv_heads != args.heads  # order-cache heads; whether the KV baseline groups query heads
    print(f"\ndecoding, d={d}, one sequence. Order cache: {Ho} heads = {H} query heads x {M} K=2 component(s), fp32. "
          f"KV baselines: {H} query heads over {H} KV heads (MHA) or {kvh} KV heads (GQA{kvh})")
    print(f"(a)/(c): mean of {args.decode_reps} back-to-back calls (CUDA events); (b)/(e): consecutive steps, "
          f"torch.cuda.synchronize around each step (perf_counter)")
    for graph in (False, True):  # load every kernel once, so one-time CUDA module loading is not timed at the first N
        cache = OrderCache(chunk=args.chunk, cuda_graph=graph)
        cache.prefill(torch.randn(Ho, 2048, 2, device="cuda"), torch.randn(Ho, 2048, d, device="cuda"))
        for _ in range(1024):
            x = torch.randn(Ho, d + 5, device="cuda")
            cache.step(x[:, :2], x[:, 2:4], x[:, 4], x[:, 5:])
        del cache
    rows = []
    for n in args.decode_lengths:
        gen = torch.Generator(device="cuda").manual_seed(n)
        g = torch.randn(Ho, n + P, 2, device="cuda", generator=gen)
        V = torch.randn(Ho, n + P, d, device="cuda", generator=gen)
        f = torch.randn(Ho, P, 2, device="cuda", generator=gen)
        b = torch.randn(Ho, P, device="cuda", generator=gen)
        r = {"n": n}
        # (c) KV-cache query baselines over n keys, eager and replayed from a CUDA graph
        K = torch.randn(H, n, d, device="cuda", generator=gen)
        Vn = torch.randn(H, n, d, device="cuda", generator=gen)
        q = torch.randn(H, 1, d, device="cuda", generator=gen)
        qb, Kb, Vb = q.bfloat16()[None], K.bfloat16()[None], Vn.bfloat16()[None]
        Kg, Vg = Kb[:, :kvh].contiguous(), Vb[:, :kvh].contiguous()
        kv_fns = {
            "kv32": (lambda x: torch.softmax(x @ K.transpose(1, 2) / d**0.5, dim=-1) @ Vn, q),
            "sdpa": (lambda x: F.scaled_dot_product_attention(x, Kb, Vb), qb),
            "gqa": (lambda x: F.scaled_dot_product_attention(x, Kg, Vg, enable_gqa=gqa), qb),
        }
        for name, (fn, x) in kv_fns.items():
            r[name] = _events(lambda: fn(x), args.decode_reps)
            replay = _graphed(fn, x.clone())
            r[name + "_g"] = _events(lambda: replay(x), args.decode_reps)
            del replay
        r["kv32_mib"], r["kvbf_mib"] = 2 * H * n * d * 4 / 2**20, 2 * H * n * d * 2 / 2**20
        r["gqa_mib"] = 2 * kvh * n * d * 2 / 2**20
        del K, Vn, Kb, Vb, Kg, Vg, kv_fns
        # (b) KV GQA bf16 decode step: write the new k / v into slot n, attend over the n + 1 keys
        Kc = torch.randn(1, kvh, n + 1, d, device="cuda", generator=gen, dtype=torch.bfloat16)
        Vc = torch.randn(1, kvh, n + 1, d, device="cuda", generator=gen, dtype=torch.bfloat16)
        new = torch.randn(P, 1, H + 2 * kvh, 1, d, device="cuda", generator=gen, dtype=torch.bfloat16)  # q, k, v

        def kv_step(x, k, v):
            Kc[:, :, n:].copy_(k)
            Vc[:, :, n:].copy_(v)
            return F.scaled_dot_product_attention(x, Kc, Vc, enable_gqa=gqa)

        replay = _graphed(kv_step, *(t.clone() for t in new[0].split([H, kvh, kvh], dim=1)))
        r["s_kv"], _ = _sync_steps(lambda i: replay(*new[i % P].split([H, kvh, kvh], dim=1)), S)
        del Kc, Vc, new, replay
        # (a), (b), (d) eager order cache
        cache = OrderCache(chunk=args.chunk)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        cache.prefill(g[:, :n], V[:, :n])
        torch.cuda.synchronize()
        r["prefill_peak"] = (torch.cuda.max_memory_allocated() - base) / 2**20
        t = time.perf_counter()
        cache.prefill(g[:, :n], V[:, :n])  # timed second prefill: the rebuild a fold costs at this size
        torch.cuda.synchronize()
        r["rebuild"], r["C"], r["mib"] = (time.perf_counter() - t) * 1000, cache.C, cache.nbytes() / 2**20
        r["q_eager"] = _events(lambda: cache.query(f[:, 0], b[:, 0]), args.decode_reps)
        r["s_eager"], _ = _steps(cache, f, g, b, V, n, S, P)
        del cache
        # (a), (d) graph order cache: the query graph and the step graphs of levels 0..8, and the pool they share
        cache = OrderCache(chunk=args.chunk, cuda_graph=True)
        cache.prefill(g[:, :n], V[:, :n])
        reserved = _reserved_after_free()
        r["q_graph"] = _events(lambda: cache.query(f[:, 0], b[:, 0]), args.decode_reps)
        _steps(cache, f, g, b, V, n, W, P)  # step W >= 256 merges into level 8
        r["pool_mib"], r["graphs"] = (_reserved_after_free() - reserved) / 2**20, cache.captures
        # (b), (d), (e) one run of max(W + S, C + 1 + W) steps from a fresh prefill (which drops the graphs)
        cache.prefill(g[:, :n], V[:, :n])
        C = cache.C
        steps = max(W + S, C + 1 + W) if args.cycle else W + S
        ts, r["fold_peak"] = _steps(cache, f, g, b, V, n, steps, P, peak_step=C if args.cycle else None)
        r["s_graph"] = ts[W:W + S]
        if args.cycle:  # steps 0..C-1 fill the tail, step C folds, the next W steps recapture the step graphs
            r["cycle"], r["fold_ms"], r["after_fold"] = ts[:C + 1], ts[C].item(), ts[C + 1:C + 1 + W]
        del cache
        # simple block-list cache (the previous implementation)
        if n <= args.simple_max_len:
            cache = OrderCacheSimple(chunk=args.chunk)
            cache.prefill(g[:, :n], V[:, :n])
            r["q_simple"] = _events(lambda: cache.query(f[:, 0], b[:, 0]), args.decode_reps)
            r["s_simple"], _ = _steps(cache, f, g, b, V, n, S, P)
            del cache
        del g, V
        torch.cuda.empty_cache()
        rows.append(r)
        print(f"  done N={n}", flush=True)

    na = "n/a"
    print(f"\n(a) query alone and (c) KV-cache query baselines, ms per query token (all heads); G = replayed from a "
          f"CUDA graph")
    print(f"{'N':>7} | {'order eager':>11} {'order G':>8} {'simple eager':>12} | {'KV fp32 MHA':>11} {'G':>6} | "
          f"{'SDPA bf16 MHA':>13} {'G':>6} | {f'SDPA bf16 GQA{kvh}':>15} {'G':>6}")
    for r in rows:
        simple = f"{r['q_simple']:>12.3f}" if "q_simple" in r else f"{na:>12}"
        print(f"{r['n']:>7} | {r['q_eager']:>11.3f} {r['q_graph']:>8.3f} {simple} | {r['kv32']:>11.3f} "
              f"{r['kv32_g']:>6.3f} | {r['sdpa']:>13.3f} {r['sdpa_g']:>6.3f} | {r['gqa']:>15.3f} {r['gqa_g']:>6.3f}")
    print(f"\n(b) per decode step, ms: mean / p50 / p99 / max over {S} consecutive steps (order: insert + merge + "
          f"query; KV: append + SDPA)")
    hdr = f"{'mean':>7}{'p50':>7}{'p99':>7}{'max':>8}"
    print(f"{'N':>7} | {'order eager (after prefill)':^29} | {f'order graph (steps {W + 1}-{W + S})':^29} | "
          f"{'simple (step 1 = cascade)':^29} | {f'KV GQA{kvh} bf16 step (graph)':^29}")
    print(f"{'':>7} | {hdr} | {hdr} | {hdr} | {hdr}")
    for r in rows:
        simple = _stats(r["s_simple"]) if "s_simple" in r else f"{na:>29}"
        print(f"{r['n']:>7} | {_stats(r['s_eager'])} | {_stats(r['s_graph'])} | {simple} | {_stats(r['s_kv'])}")
    deep = [j for j in range(64) if W < (1 << j) <= W + S and j > 8]
    print(f"    no fold falls in these windows when C >= {W + S}; graph steps merging into a level > 8 run eagerly "
          f"(here {', '.join(f'level {j} at step {1 << j}' for j in deep) or 'none'})")
    print("\n(d) memory, MiB. Order cache steady state = buffers (nbytes) + the pool its CUDA graphs share; peaks = "
          "max allocated during a prefill / the fold step (buffers included)")
    print(f"{'N':>7} | {'buffers':>8} {'graph pool':>10} {'total':>7} | {'prefill peak':>12} {'fold peak':>9} | "
          f"{'KV fp32 MHA':>11} {'KV bf16 MHA':>11} {f'KV bf16 GQA{kvh}':>14} | {'x bf16 MHA':>10} "
          f"{f'x bf16 GQA{kvh}':>13} | {'tail C':>7}")
    for r in rows:
        total = r["mib"] + r["pool_mib"]
        fold = f"{r['fold_peak']:>9.0f}" if r["fold_peak"] is not None else f"{na:>9}"
        print(f"{r['n']:>7} | {r['mib']:>8.0f} {r['pool_mib']:>10.0f} {total:>7.0f} | {r['prefill_peak']:>12.0f} "
              f"{fold} | {r['kv32_mib']:>11.0f} {r['kvbf_mib']:>11.0f} {r['gqa_mib']:>14.0f} | "
              f"{total / r['kvbf_mib']:>10.2f} {total / r['gqa_mib']:>13.2f} | {r['C']:>7}")
    print(f"    graph pool = reserved-memory growth over the {rows[0]['graphs']} captures (query + step levels 0..8); "
          f"a fold drops the graphs and recaptures them into a fresh pool")
    if args.cycle:
        print("\n(e) one full tail cycle, CUDA graph: the C + 1 steps from prefill to the first fold (incl. 10 graph "
              "captures, all merges, the fold)")
        print(f"{'N':>7} | {'steps':>6} {'mean ms':>8} {'p50':>7} {'p99':>7} | {'fold step ms':>12} "
              f"{'prefill ms':>10} | {f'next {W} steps: mean / max':>26}")
        for r in rows:
            c, a = r["cycle"], r["after_fold"]
            nxt = f"{a.mean():>12.3f} / {a.max():>9.2f}" if len(a) else f"{na:>26}"
            print(f"{r['n']:>7} | {len(c):>6} {c.mean():>8.3f} {c.quantile(0.5):>7.3f} {c.quantile(0.99):>7.3f} | "
                  f"{r['fold_ms']:>12.1f} {r['rebuild']:>10.1f} | {nxt}")
        print("    prefill ms = a timed eager prefill of N tokens (the rebuild part of a fold); the fold step also "
              "reallocates the buffer and captures the query graph; the next steps recapture the step graphs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--kv-heads", type=int, default=8, help="KV heads of the GQA baseline (Qwen3-0.6B / 1.7B: 8)")
    p.add_argument("--components", type=int, default=1,
                   help="K=2 order components per query head (decode): the order cache holds heads x components heads")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--decode-lengths", type=int, nargs="+", default=[1024, 4096, 16384, 65536, 262144])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--decode-reps", type=int, default=200)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--graph-warmup", type=int, default=256)
    p.add_argument("--simple-max-len", type=int, default=65536)
    p.add_argument("--pool", type=int, default=4096, help="fresh random tokens the decode steps cycle through")
    p.add_argument("--no-cycle", dest="cycle", action="store_false", help="skip (e), the full tail cycle")
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
