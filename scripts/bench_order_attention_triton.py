"""Benchmark the Triton causal order attention (learning/order_attention_triton.py), forward and forward + backward.

One layer's attention output for B sequences, H heads, head dim d, for a single K = 2 order head per head (M = 1,
``order_attention_triton``) and a gated mixture of M of them (``mixture_attention_triton``, sharing V): the Triton
kernels (chunk 64, and the ``--extra-chunks`` at the ``--extra-chunk-length``) against the torch position-tree
``order_attention`` / ``mixture_attention`` (up to ``--torch-max-len``; autograd through it for the backward, up to
``--torch-train-max-len``) and a standard causal dot-product head: PyTorch SDPA in fp32 (default dispatch) and in bf16
on each fast backend, FlashAttention-2 (``SDPBackend.FLASH_ATTENTION``, what the default dispatch picks) and cuDNN
(``SDPBackend.CUDNN_ATTENTION``, about 1.8x faster on a GH200), plus FlashAttention-3 (``flash_attn_interface``) when it
is installed. The SDPA columns do not depend on M (one dot-product head per head). The last column is the speedup over
the fastest bf16 backend.

Forward rows time ``attention(...)`` under no_grad; forward + backward rows time ``out = attention(...);
out.backward(dO)`` with every input requiring grad (f, g, b, gates, V; q, k, v). Times are CUDA-event means over
``--reps`` back-to-back calls after a warm-up (which includes Triton compilation); memory is the peak allocated during
one call above what was allocated before it (outputs, saved tensors and input gradients included).

Then per-kernel breakdowns at ``--breakdown-length`` from the torch profiler: CUDA time per forward of the segmented
sorts, the scan, the search and the query kernels and of the remaining glue (rank differences, padding, casts); per
backward of the prep, sorts, scan, search, intra and tree kernels and the glue; and per mixture forward + backward (the
largest of ``--mixtures``, if above 1), where the glue is what the fused mixture leaves besides the kernels (softmax,
the gates' softmax backward, gradient sums and casts).

Usage (repo root, GPU node): python scripts/bench_order_attention_triton.py [--lengths 2048 8192 32768 65536]
"""
import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from flash_attn_interface import flash_attn_func as fa3_func
except ImportError:
    fa3_func = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from learning.order_attention import mixture_attention, order_attention  # noqa: E402
from learning.order_attention_triton import (  # noqa: E402
    _order_attention_bwd, _order_attention_fwd, mixture_attention_triton, order_attention_triton,
)

NAN = (float("nan"), float("nan"))


def timed(fn, reps, warmup=2):
    """(mean ms per call over ``reps`` back-to-back calls, peak extra MiB of one call)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    mem = (torch.cuda.max_memory_allocated() - base) / 2**20
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps, mem


def guarded(fn, *args, **kwargs):
    """``timed``, or nan when the call runs out of memory or the backend refuses the inputs."""
    try:
        return timed(fn, *args, **kwargs)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        torch.cuda.empty_cache()
        return NAN


def _inputs(B, H, n, d, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    f = torch.randn(B, H, n, 2, device="cuda", generator=gen) * 2
    g = torch.randn(B, H, n, 2, device="cuda", generator=gen) * 2
    b = torch.randn(B, H, n, device="cuda", generator=gen) * 2
    V = torch.randn(B, H, n, d, device="cuda", generator=gen)
    return f, g, b, V


def _fmt(t, m):
    return f"{t:>13.2f}{m:>7.0f}" if t == t else f"{'-':>13}{'-':>7}"


def _step(fn, leaves, dO, train):
    """A no-grad forward, or a forward + backward into the (cleared) gradients of ``leaves``."""
    if not train:
        def run():
            with torch.no_grad():
                fn()
        return run

    def run():
        for t in leaves:
            t.grad = None
        fn().backward(dO)
    return run


def _sdpa_rows(n, args, train):
    """Causal SDPA fp32 (default dispatch) and bf16 on FA2 / cuDNN (/ FA3) for random q, k, v of one length."""
    B, H, d = args.batch, args.heads, args.dim
    q32, k32, v32, dO = (torch.randn(B, H, n, d, device="cuda") for _ in range(4))
    q16, k16, v16 = (x.bfloat16() for x in (q32, k32, v32))
    if train:
        for x in (q32, k32, v32, q16, k16, v16):
            x.requires_grad_()
    rows = []
    for (q, k, v), backend in [((q32, k32, v32), None), ((q16, k16, v16), SDPBackend.FLASH_ATTENTION),
                               ((q16, k16, v16), SDPBackend.CUDNN_ATTENTION)]:
        fn = _step(lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v, is_causal=True), (q, k, v),
                   dO.to(q.dtype), train)
        if backend is None:
            rows.append(guarded(fn, args.reps, warmup=1))
        else:
            with sdpa_kernel(backend):
                rows.append(guarded(fn, args.reps))
    if fa3_func:
        qt, kt, vt = (x.detach().transpose(1, 2).contiguous().requires_grad_(train) for x in (q16, k16, v16))
        dOt = dO.bfloat16().transpose(1, 2).contiguous()

        def fa3():
            out = fa3_func(qt, kt, vt, causal=True)
            return out[0] if isinstance(out, tuple) else out
        rows.append(guarded(_step(fa3, (qt, kt, vt), dOt, train), args.reps))
    return rows


def bench(args, train):
    B, H, d = args.batch, args.heads, args.dim
    what = "forward + backward" if train else "forward"
    print(f"\n{what}, B={B} H={H} d={d}; order attention fp32 (f, g, b, gates, V fp32); ms and peak extra MiB per call")
    sdpa = {}
    for M in args.mixtures:
        cols = ["triton C=64", "torch tree", "SDPA fp32", "FA2 bf16", "cuDNN bf16"] + (["FA3 bf16"] if fa3_func else [])
        print(f"{'N':>6} {'M':>2} | " + " | ".join(f"{c + ' ms':>13}{'MiB':>7}" for c in cols)
              + f" | {'vs tree':>8} | {'vs best bf16':>12}")
        extra = []
        for n in args.lengths:
            comps = [_inputs(B, H, n, d, seed=n + m)[:3] for m in range(M)]
            V = _inputs(B, H, n, d, seed=n)[3]
            gates = torch.randn(B, H, n, M, device="cuda")
            dO = torch.randn(B, H, n, d, device="cuda")
            leaves = [V] + ([gates] if M > 1 else []) + [t for c in comps for t in c]
            if train:
                for t in leaves:
                    t.requires_grad_()
            if M == 1:
                tri = lambda c=args.chunk, V=V: order_attention_triton(*comps[0], V, c)  # noqa: E731
                tree = lambda: order_attention(*comps[0], V, True, args.chunk)  # noqa: E731
            else:
                tri = lambda c=args.chunk, V=V: mixture_attention_triton(gates, comps, V, c)  # noqa: E731
                tree = lambda: mixture_attention(gates, comps, V, True, args.chunk)  # noqa: E731
            res = [guarded(_step(tri, leaves, dO, train), args.reps)]
            tmax = args.torch_train_max_len if train else args.torch_max_len
            res.append(guarded(_step(tree, leaves, dO, train), max(1, args.reps // 4), warmup=1) if n <= tmax else NAN)
            if (n, train) not in sdpa:
                sdpa[n, train] = _sdpa_rows(n, args, train)
            res += sdpa[n, train]
            tt, to = res[0][0], res[1][0]
            best16 = min((t for t, _ in res[3:] if t == t), default=float("nan"))
            vs_tree = f"{to / tt:>7.1f}x" if to == to else f"{'-':>8}"
            print(f"{n:>6} {M:>2} | " + " | ".join(_fmt(t, m) for t, m in res)
                  + f" | {vs_tree} | {best16 / tt:>11.2f}x", flush=True)
            if M == 1 and n == args.extra_chunk_length:
                for c in args.extra_chunks:
                    extra.append((n, c, *guarded(_step(lambda c=c: tri(c), leaves, dO, train), args.reps)))
            if M == 1 and (n == args.extra_chunk_length or n == args.lengths[-1]):
                V16 = V.detach().bfloat16().requires_grad_(train)
                lv16 = [V16] + leaves[1:]
                extra.append((n, f"{args.chunk}, V bf16",
                              *guarded(_step(lambda: tri(V=V16), lv16, dO, train), args.reps)))
            del comps, V, gates, dO, leaves
            torch.cuda.empty_cache()
        for n, c, t, m in extra:
            print(f"triton chunk {c} at N={n}: {t:.2f} ms, {m:.0f} MiB")


def _category(name):
    low = name.lower()
    for kern, cat in [("_bwd_prep_kernel", "prep (Triton)"), ("_bwd_scan_kernel", "scan (Triton)"),
                      ("_bwd_search_kernel", "search (Triton)"), ("_key_intra_kernel", "intra (Triton)"),
                      ("_key_tree_kernel", "tree (Triton)"), ("_scan_kernel", "scan (Triton)"),
                      ("_search_kernel", "search (Triton)"), ("_query_kernel", "query (Triton)"),
                      ("_mix_sum_kernel", "mixture sum (Triton)")]:
        if kern in name:
            return cat
    if "sort" in low or "radix" in low:
        return "sorts (torch.sort)"
    return "glue (elementwise, pad, copies)"


def _breakdown(fn, reps, cats, title):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    tot = defaultdict(float)
    calls = defaultdict(int)
    for ev in prof.key_averages():
        if ev.device_type != torch.autograd.DeviceType.CUDA:  # kernels only (CPU ops would count them twice)
            continue
        cat = _category(ev.key)
        tot[cat] += ev.device_time_total / reps / 1000
        calls[cat] += ev.count // reps
    wall, _ = timed(fn, reps)
    print(f"\n{title}")
    for cat in cats:
        print(f"  {cat:<34}{tot[cat]:>8.3f} ms  ({calls[cat]} kernels)")
    print(f"  {'sum of kernels':<34}{sum(tot.values()):>8.3f} ms;  wall per call (CUDA events) {wall:.3f} ms")


@torch.no_grad()
def bench_breakdown(args):
    B, H, d, n = args.batch, args.heads, args.dim, args.breakdown_length
    f, g, b, V = _inputs(B, H, n, d, seed=1)
    dO = torch.randn(B, H, n, d, device="cuda")
    glue = "glue (elementwise, pad, copies)"
    for chunk in [args.chunk] + [c for c in args.extra_chunks if c != args.chunk]:
        _breakdown(lambda: _order_attention_fwd(f, g, b, V, chunk), args.reps,
                   ["sorts (torch.sort)", "scan (Triton)", "search (Triton)", "query (Triton)", glue],
                   f"forward per-kernel breakdown, N={n} B={B} H={H} d={d} chunk={chunk} (GPU ms per call, profiler)")
        saved = _order_attention_fwd(f, g, b, V, chunk)
        _breakdown(lambda: _order_attention_bwd(f, g, b, V, *saved, dO, chunk), args.reps,
                   ["prep (Triton)", "sorts (torch.sort)", "scan (Triton)", "search (Triton)", "intra (Triton)",
                    "tree (Triton)", glue],
                   f"backward per-kernel breakdown, N={n} B={B} H={H} d={d} chunk={chunk} (GPU ms per call, profiler)")
        del saved
    M = max(args.mixtures)
    if M > 1:
        comps = [tuple(t.requires_grad_() for t in _inputs(B, H, n, d, seed=n + m)[:3]) for m in range(M)]
        gates = torch.randn(B, H, n, M, device="cuda").requires_grad_()
        V.requires_grad_()
        leaves = [gates, V] + [t for c in comps for t in c]

        def mix_step():
            for t in leaves:
                t.grad = None
            with torch.enable_grad():
                mixture_attention_triton(gates, comps, V, args.chunk).backward(dO)
        _breakdown(mix_step, args.reps,
                   ["prep (Triton)", "sorts (torch.sort)", "scan (Triton)", "search (Triton)", "query (Triton)",
                    "intra (Triton)", "tree (Triton)", "mixture sum (Triton)", glue],
                   f"mixture forward + backward per-kernel breakdown, M={M} N={n} B={B} H={H} d={d} chunk={args.chunk}"
                   " (GPU ms per call, profiler)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 32768, 65536])
    p.add_argument("--mixtures", type=int, nargs="+", default=[1, 4], help="order heads per head (M > 1: mixture)")
    p.add_argument("--extra-chunks", type=int, nargs="*", default=[128])
    p.add_argument("--extra-chunk-length", type=int, default=32768)
    p.add_argument("--torch-max-len", type=int, default=32768, help="skip the torch tree forward above this length")
    p.add_argument("--torch-train-max-len", type=int, default=8192,
                   help="skip the torch tree forward + backward above this length (autograd memory)")
    p.add_argument("--breakdown-length", type=int, default=32768)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--no-forward", action="store_true")
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--no-breakdown", action="store_true")
    args = p.parse_args()
    print(f"device: {torch.cuda.get_device_name()}, torch {torch.__version__}")
    if not args.no_forward:
        bench(args, train=False)
    if not args.no_train:
        bench(args, train=True)
    if not args.no_breakdown:
        bench_breakdown(args)


if __name__ == "__main__":
    main()
