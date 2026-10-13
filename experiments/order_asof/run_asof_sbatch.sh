#!/bin/bash
#SBATCH --job-name=order_asof
#SBATCH --partition=debug
#SBATCH --account=a0087
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=780G
#SBATCH --time=01:30:00
#SBATCH --output=/capstor/store/cscs/swissai/a0087/tianyu/partial/logs/order_asof_%j.log

# Temporal as-of task on Qwen3-8B (YaRN x4, bf16), last 6 layers; train.pt = 1500 docs x 16k, eval 48 docs at
# 32k / 64k / 128k. One lane per GPU:
#   0: base full attention (zero-shot) then base + sliding window W=4096 and W=16384 (training-free)
#   1: order1 student (Triton), distilled 300 steps then 600 task steps
#   2: rope128s student, full attention, same training
#   3: rope128s student trained and evaluated with the W=4096 window (the cost-matched control)
set -euo pipefail
ROOT=/capstor/store/cscs/swissai/a0087/tianyu
DATA=$ROOT/partial/order_asof/data
OUT=${OUT:-$ROOT/partial/order_asof/runs/${SLURM_JOB_ID}}
REPO=/users/tliu/repos/partial
STEPS=${STEPS:-600}
DISTILL=${DISTILL:-300}
export HF_HOME=$ROOT/hf_home HF_HUB_CACHE=$ROOT/hf_home/hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$REPO"
mkdir -p "$OUT"
PY="$REPO/.venv/bin/python $REPO/experiments/order_asof/train_asof.py --model Qwen/Qwen3-8B --data $DATA --out $OUT \
    --eval 32768 65536 131072"
run() {
  local log=$1; shift
  srun --exclusive --ntasks=1 --gpus=1 --cpus-per-task=72 --mem=190G --uenv=pytorch/v2.9.1:v2 --view=default \
       bash -c "ulimit -c 0; $*" > "$log" 2>&1
}
run "$OUT/lane0.log" "$PY --arm base; $PY --arm base --window 4096; $PY --arm base --window 16384" &
run "$OUT/lane1.log" $PY --arm order1 --train --distill-steps $DISTILL --steps $STEPS &
run "$OUT/lane2.log" $PY --arm rope128s --train --distill-steps $DISTILL --steps $STEPS &
run "$OUT/lane3.log" $PY --arm rope128s --window 4096 --train --distill-steps $DISTILL --steps $STEPS &
wait
grep -h "^eval" "$OUT"/lane*.log
