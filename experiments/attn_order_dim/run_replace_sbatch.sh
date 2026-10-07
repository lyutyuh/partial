#!/bin/bash
#SBATCH --job-name=attn_replace
#SBATCH --partition=debug
#SBATCH --account=a0087
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=780G
#SBATCH --time=01:30:00
#SBATCH --output=/capstor/store/cscs/swissai/a0087/tianyu/partial/logs/attn_replace_%j.log

# KL-fitted order scorers vs attention heads (replace.py), Qwen3-0.6B/1.7B-Base on the extract.py data:
#   GPU 0/1: layers 22-27, all scorers, head replacement -> WikiText-103 test perplexity
#   GPU 2/3: all 28 layers, exact-fast scorers only, recall@k of the teacher's argmax (indexer use)
set -euo pipefail
ROOT=/capstor/store/cscs/swissai/a0087/tianyu/partial
OUT=$ROOT/attn_order_dim
REPO=/users/tliu/repos/partial
STEPS=${STEPS:-2000}
export HF_HOME=$ROOT/hf_cache HF_HUB_CACHE=$ROOT/hf_cache/hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$REPO"
mkdir -p "$OUT/replace"
run() {
  local log=$1; shift
  srun --exclusive --ntasks=1 --gpus=1 --cpus-per-task=72 --mem=190G --uenv=pytorch/v2.9.1:v2 --view=default \
       bash -c "ulimit -c 0; $*" > "$log" 2>&1
}
for m in 0.6B 1.7B; do
  run "$OUT/replace/replace_${m}_${SLURM_JOB_ID}.log" .venv/bin/python experiments/attn_order_dim/replace.py \
      --model Qwen/Qwen3-${m}-Base --data "$OUT/data/qwen3-${m}" --layers 22 23 24 25 26 27 --steps $STEPS \
      --scorers order:2 osum:2 osum:3 omix:4 dot:2 dot:8 dot:128 --perplexity --layer-sets 27 26-27 24-27 22-27 \
      --out "$OUT/replace/qwen3-${m}_replace_${SLURM_JOB_ID}.jsonl" &
  run "$OUT/replace/recall_${m}_${SLURM_JOB_ID}.log" .venv/bin/python experiments/attn_order_dim/replace.py \
      --model Qwen/Qwen3-${m}-Base --data "$OUT/data/qwen3-${m}" --layers $(seq 0 27 | tr '\n' ' ') --steps $STEPS \
      --scorers order:2 osum:2 osum:3 dot:2 \
      --out "$OUT/replace/qwen3-${m}_recall_${SLURM_JOB_ID}.jsonl" &
done
wait
