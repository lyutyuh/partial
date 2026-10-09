#!/bin/bash
#SBATCH --job-name=attn_recall
#SBATCH --partition=debug
#SBATCH --account=a0087
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=780G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/store/cscs/swissai/a0087/tianyu/partial/logs/attn_recall_%j.log

# Recall re-run (longctx.py --stages recall) on already-trained order1 students: pooled non-sink recall@k, the
# learning-free window baseline (sink + k-1 most recent keys) and the union indexer (student top k/2 + window top
# k/2), also restricted to non-trivial queries (more than k visible keys). Students are copied from SRC.
#   GPU 0/1: 0.6B / 1.7B at 512, 2048, 8192;  GPU 2/3: 0.6B / 1.7B at 32768 (all 9 windows)
set -euo pipefail
ROOT=/capstor/store/cscs/swissai/a0087/tianyu/partial
SRC=${SRC:-$ROOT/attn_order_dim/longctx/3618191}
OUT=${OUT:-$ROOT/attn_order_dim/recall/${SLURM_JOB_ID}}
REPO=/users/tliu/repos/partial
export HF_HOME=$ROOT/hf_cache HF_HUB_CACHE=$ROOT/hf_cache/hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$REPO"
mkdir -p "$OUT"
cp "$SRC"/qwen3-*_students_order1.pt "$OUT"/
PY="$REPO/.venv/bin/python $REPO/experiments/attn_order_dim/longctx.py --out $OUT --stages recall --scorers order1 \
    --recall-ks 1 8 64 256"
run() {
  local log=$1; shift
  srun --exclusive --ntasks=1 --gpus=1 --cpus-per-task=72 --mem=190G --uenv=pytorch/v2.9.1:v2 --view=default \
       bash -c "ulimit -c 0; $*" > "$log" 2>&1
}
for m in 0.6B 1.7B; do
  run "$OUT/short_${m}.log" $PY --model Qwen/Qwen3-${m}-Base --part short --recall-lengths 512 2048 8192 &
  run "$OUT/long_${m}.log" $PY --model Qwen/Qwen3-${m}-Base --part long --recall-lengths 32768 &
done
wait
for m in 0.6B 1.7B; do
  run "$OUT/merge_${m}.log" $PY --model Qwen/Qwen3-${m}-Base --merge
  cat "$OUT/merge_${m}.log"
done
