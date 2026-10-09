#!/bin/bash
#SBATCH --job-name=attn_finetune
#SBATCH --partition=debug
#SBATCH --account=a0087
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=780G
#SBATCH --time=01:30:00
#SBATCH --output=/capstor/store/cscs/swissai/a0087/tianyu/partial/logs/attn_finetune_%j.log

# End-to-end fine-tuning of the replaced layers 22-27 (longctx.py --stages finetune): students warm-started from the
# KL-distilled ones in SRC are trained on KL(base || replaced) of the model's next-token distribution, on FT_LEN-token
# windows of the WikiText-103 TRAIN split (disjoint from the test stream), order scorers through the Triton kernels
# and the matched dot-product control (rope128s) through SDPA, identical data / steps / lr / warm start for all.
# Then perplexity on the test stream at 512..32768 (same tokens at every N), order scorers through --impl triton.
#   GPU 0/2 (0.6B / 1.7B): order1 then omix4 -> ppl (no base);  GPU 1/3: rope128s -> ppl with base
set -euo pipefail
ROOT=/capstor/store/cscs/swissai/a0087/tianyu/partial
SRC=${SRC:-$ROOT/attn_order_dim/longctx/3618191}
OUT=${OUT:-$ROOT/attn_order_dim/finetune/${SLURM_JOB_ID}}
REPO=/users/tliu/repos/partial
FT_STEPS=${FT_STEPS:-300}
FT_LEN=${FT_LEN:-16384}
FT_LR=${FT_LR:-3e-4}
export HF_HOME=$ROOT/hf_cache HF_HUB_CACHE=$ROOT/hf_cache/hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$REPO"
mkdir -p "$OUT"
PY="$REPO/.venv/bin/python $REPO/experiments/attn_order_dim/longctx.py --out $OUT --init-from $SRC --impl triton \
    --ft-steps $FT_STEPS --ft-len $FT_LEN --ft-lr $FT_LR"
run() {
  local log=$1; shift
  srun --exclusive --ntasks=1 --gpus=1 --cpus-per-task=72 --mem=190G --uenv=pytorch/v2.9.1:v2 --view=default \
       bash -c "ulimit -c 0; $*" > "$log" 2>&1
}
for m in 0.6B 1.7B; do
  run "$OUT/order_${m}.log" $PY --model Qwen/Qwen3-${m}-Base --part ft_order --stages finetune ppl \
      --scorers order1 omix4 --subset-scorers order1 omix4 --no-base &
  run "$OUT/rope_${m}.log" $PY --model Qwen/Qwen3-${m}-Base --part ft_rope --stages finetune ppl \
      --scorers rope128s --subset-scorers &
done
wait
for m in 0.6B 1.7B; do
  run "$OUT/merge_${m}.log" $PY --model Qwen/Qwen3-${m}-Base --merge
  cat "$OUT/merge_${m}.log"
done
