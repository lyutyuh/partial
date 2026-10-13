"""Feasibility and speed of order attention in Qwen3-8B at 32k-128k tokens (YaRN rope scaling, bf16, batch 1).

Measures, for a full backbone forward (no grad, lm_head excluded, batch 1): the base model with the causal fast path
(cuDNN / FlashAttention-2 SDPA) and the same model with its last R layers' attention replaced by order attention
through the Triton kernels (randomly initialised order students: cost only, not quality), plus peak memory. Also the
per-layer attention module cost at 128k. The explicit HF mask is dropped so SDPA runs is_causal (exact for batch 1).

Usage: python feasibility_8b.py [--model Qwen/Qwen3-8B] [--lengths 32768 65536 131072] [--replace 6] [--yarn 4]
"""
import argparse
import contextlib
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "/users/tliu/repos/partial/experiments/attn_order_dim")
sys.path.insert(0, "/users/tliu/repos/partial")
import longctx  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402


def timed(fn, reps):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


@contextlib.contextmanager
def causal_without_mask(model):
    def drop(mod, args, kwargs):
        kwargs["attention_mask"] = None
        return args, kwargs
    hooks = [layer.self_attn.register_forward_pre_hook(drop, with_kwargs=True) for layer in model.model.layers]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--lengths", type=int, nargs="+", default=[32768, 65536, 131072])
    p.add_argument("--replace", type=int, default=6)
    p.add_argument("--yarn", type=float, default=4.0)
    p.add_argument("--reps", type=int, default=2)
    args = p.parse_args()
    cfg = AutoConfig.from_pretrained(args.model)
    if args.yarn > 1:
        cfg.rope_scaling = {"rope_type": "yarn", "factor": args.yarn, "original_max_position_embeddings": 32768}
        cfg.max_position_embeddings = int(32768 * args.yarn)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=cfg, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    L = cfg.num_hidden_layers
    layers = list(range(L - args.replace, L))
    students = {l: longctx.Student(cfg.hidden_size, cfg.num_attention_heads, "order1", hidden=1024,
                                   rope_dim=cfg.head_dim).cuda().eval() for l in layers}
    with torch.no_grad():  # realistic magnitudes for the cost run: small random ranks, mild slopes
        for st in students.values():
            st.slope.uniform_(-0.3, 0.3)
    print(f"{args.model}: {L} layers, H={cfg.num_attention_heads}, kv={cfg.num_key_value_heads}, d={cfg.head_dim}, "
          f"yarn x{args.yarn}; replacing layers {layers[0]}-{layers[-1]} ({len(layers)}) with order1 (Triton)")
    print(f"{'N':>7} | {'base FA2 ms':>11} {'base cuDNN ms':>13} {'peak GB':>8} | {'order ms':>9} {'peak GB':>8} | "
          f"{'whole model':>11} | {'per replaced layer: cuDNN attn ms -> order attn ms':>50}", flush=True)
    for N in args.lengths:
        ids = torch.randint(0, 1000, (1, N), device="cuda")
        run = lambda: model.model(input_ids=ids, use_cache=False)  # noqa: E731
        with causal_without_mask(model):
            torch.cuda.reset_peak_memory_stats()
            t_fa2 = timed(run, args.reps)
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                t_cudnn = timed(run, args.reps)
            pk_base = torch.cuda.max_memory_allocated() / 2**30
            # per-layer attention module cost: time one unreplaced layer's self_attn on captured inputs
            cap = {}
            h = model.model.layers[layers[0]].self_attn.register_forward_pre_hook(
                lambda m, a, k: cap.__setitem__("x", (a, k)), with_kwargs=True)
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                run()
            h.remove()
            a_, k_ = cap["x"]
            attn = model.model.layers[layers[0]].self_attn
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                t_attn = timed(lambda: attn(*a_, **k_), args.reps)
            torch.cuda.reset_peak_memory_stats()
            with longctx.replaced(model, students, layers, impl="triton"), sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                t_order = timed(run, args.reps)
                rep = model.model.layers[layers[0]].self_attn
                t_oattn = timed(lambda: rep(*a_, **k_), args.reps)
            pk_order = torch.cuda.max_memory_allocated() / 2**30
        print(f"{N:>7} | {t_fa2:>11.0f} {t_cudnn:>13.0f} {pk_base:>8.1f} | {t_order:>9.0f} {pk_order:>8.1f} | "
              f"{100 * (t_order / t_cudnn - 1):>+10.1f}% | {t_attn:>8.1f} -> {t_oattn:.1f} ms "
              f"({t_attn / t_oattn:.1f}x per layer)", flush=True)
        del ids, cap
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
