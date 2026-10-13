#!/bin/bash
#SBATCH --job-name=order_oracle
#SBATCH --partition=debug
#SBATCH --account=a0087
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=780G
#SBATCH --time=01:30:00
#SBATCH --output=/capstor/store/cscs/swissai/a0087/tianyu/partial/logs/order_oracle_%j.log

# Oracle-head test on Qwen3-8B (YaRN x4, bf16): one exact as-of order head (realizers from the tokens, the layer's
# own values) added to ONE layer through a zero-initialised linear adapter trained 300 steps on the 16k training docs;
# evaluated at 32k / 64k / 128k. One lane per GPU:
#   0: kernel retrieval check, then oracle head at layer 30
#   1: 'latest' control at layer 30 (right entity and band, time threshold past the end of the log)
#   2: 'selfv' control at layer 30 (same adapter on the position's own value vectors, no retrieval)
#   3: oracle head at layer 18 (more layers left to decode the retrieved value)
set -euo pipefail
ROOT=/capstor/store/cscs/swissai/a0087/tianyu
DATA=$ROOT/partial/order_asof/data
OUT=${OUT:-$ROOT/partial/order_asof/runs/${SLURM_JOB_ID}}
REPO=/users/tliu/repos/partial
STEPS=${STEPS:-300}
LR=${LR:-1e-3}
export HF_HOME=$ROOT/hf_home HF_HUB_CACHE=$ROOT/hf_home/hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$REPO"
mkdir -p "$OUT"
PY="$REPO/.venv/bin/python $REPO/experiments/order_asof/train_asof.py --model Qwen/Qwen3-8B --data $DATA --out $OUT \
    --eval 32768 65536 131072 --train --steps $STEPS --lr $LR"
CHECK="$REPO/.venv/bin/python $REPO/experiments/order_asof/oracle_kernel_check.py --docs 4"
run() {
  local log=$1; shift
  srun --exclusive --ntasks=1 --gpus=1 --cpus-per-task=72 --mem=190G --uenv=pytorch/v2.9.1:v2 --view=default \
       bash -c "ulimit -c 0; $*" > "$log" 2>&1
}
run "$OUT/lane0.log" "$CHECK; $PY --arm oracle --replace 6 --tag oracle_L30" &
run "$OUT/lane1.log" $PY --arm oracle_latest --replace 6 --tag latest_L30 &
run "$OUT/lane2.log" $PY --arm oracle_selfv --replace 6 --tag selfv_L30 &
run "$OUT/lane3.log" $PY --arm oracle --replace 18 --tag oracle_L18 &
wait
grep -h "^N=\|^eval" "$OUT"/lane*.log
