"""Training-free order-realizability prediction from a head's weights: spectrum of its effective bilinear form.

For each head, recompute q = RoPE(q_norm(W_q x)) and k = RoPE(k_norm(W_k x)) on the extracted attention inputs x, then
measure how concentrated the score function s(x, y) = q(x)^T k(y) is in a few directions:
  * weight spectrum:  singular values of W_q^T W_k  (per head, ignores norm/RoPE/input distribution)
  * data spectrum:    singular values of the cross-covariance E[q k^T] over tokens (what the score function uses)
  * score-rank:       fraction of the variance of s(x, y) over random (x, y) pairs captured by the rank-r truncation
The energy in the top r directions is the Eckart-Young optimal rank-r approximation error of the bilinear form, i.e.
how well r summed K = 2 order heads (or a rank-r dot product) could in principle reproduce the head. Compare against the
fitted accuracies from fit.py to see whether realizability is a property of the weights or of the inputs.

Writes one json line per head to --out. Usage: python spectrum.py --model Qwen/Qwen3-0.6B-Base --data <extract dir>
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-seq", type=int, default=64)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4, 8, 16, 32, 64])
    args = p.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    cfg = model.config
    H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    rot = model.model.rotary_emb
    with open(args.out, "w") as fh:
        for l, layer in enumerate(model.model.layers):
            d = torch.load(os.path.join(args.data, "train", f"layer{l:02d}.pt"))
            x = d["x"][: args.n_seq].float()  # (S, N, d_model): the layer's attention input
            S, N, _ = x.shape
            pos = torch.arange(N)[None].expand(S, -1)
            cos, sin = rot(x, pos)
            at = layer.self_attn
            q = at.q_norm(at.q_proj(x).view(S, N, H, D)).transpose(1, 2)
            k = at.k_norm(at.k_proj(x).view(S, N, Hkv, D)).transpose(1, 2)
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = k.repeat_interleave(H // Hkv, dim=1)
            Wq = at.q_proj.weight.view(H, D, -1)
            Wk = at.k_proj.weight.view(Hkv, D, -1).repeat_interleave(H // Hkv, dim=0)
            for h in range(H):
                qh = q[:, h].reshape(-1, D)
                kh = k[:, h].reshape(-1, D)
                qh = qh - qh.mean(0)
                kh = kh - kh.mean(0)
                # weight spectrum
                sw = torch.linalg.svdvals(Wq[h].T @ Wk[h])
                ew = (sw ** 2).cumsum(0) / (sw ** 2).sum()
                # data spectrum of the cross-covariance
                C = qh.T @ kh / qh.shape[0]
                U, sd, Vt = torch.linalg.svd(C)
                ed = (sd ** 2).cumsum(0) / (sd ** 2).sum()
                # score-rank: variance of s over random pairs explained by rank-r truncation of the score function
                idx = torch.randint(0, qh.shape[0], (4096,))
                jdx = torch.randint(0, kh.shape[0], (4096,))
                s_full = (qh[idx] * kh[jdx]).sum(-1)
                # rank-r truncation: s_r = q^T U_r U_r^T k, the score restricted to the top-r left-singular directions of
                # the cross-covariance (the subspace where queries and keys actually co-vary)
                a = qh[idx] @ U
                b = kh[jdx] @ U
                var_full = s_full.var()
                sr = {}
                for r in args.ranks:
                    s_r = (a[:, :r] * b[:, :r]).sum(-1)
                    sr[r] = float(1 - (s_full - s_r).var() / var_full)
                rec = {"layer": l, "head": h,
                       "weight_energy": {r: float(ew[r - 1]) for r in args.ranks},
                       "data_energy": {r: float(ed[r - 1]) for r in args.ranks},
                       "score_var_explained": sr,
                       "q_rms": float(qh.norm(dim=-1).mean()), "k_rms": float(kh.norm(dim=-1).mean())}
                fh.write(json.dumps(rec) + "\n")
            print(f"layer {l}: last head weight top-1 energy {ew[0].item():.3f}, data top-1 {ed[0].item():.3f}", flush=True)


if __name__ == "__main__":
    main()
