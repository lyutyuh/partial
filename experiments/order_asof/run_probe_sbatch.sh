#!/bin/bash
#SBATCH --job-name=order_probe
#SBATCH --partition=debug
#SBATCH --account=a0087
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=780G
#SBATCH --time=00:40:00
#SBATCH --output=/capstor/store/cscs/swissai/a0087/tianyu/partial/logs/order_probe_%j.log

# Realizer probes on frozen Qwen3-8B hidden states (time_probe.py): can a probe read a record's time and the
# watched-entity gate precisely enough for a tent as-of head? Lanes: key position x fit size.
set -euo pipefail
ROOT=/capstor/store/cscs/swissai/a0087/tianyu
OUT=${OUT:-$ROOT/partial/order_asof/runs/${SLURM_JOB_ID}}
REPO=/users/tliu/repos/partial
export HF_HOME=$ROOT/hf_home HF_HUB_CACHE=$ROOT/hf_home/hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$REPO"
mkdir -p "$OUT"
PY="$REPO/.venv/bin/python $REPO/experiments/order_asof/time_probe.py --out $OUT --layers 12 18 24 30"
run() {
  local log=$1; shift
  srun --exclusive --ntasks=1 --gpus=1 --cpus-per-task=72 --mem=190G --uenv=pytorch/v2.9.1:v2 --view=default \
       bash -c "ulimit -c 0; $*" > "$log" 2>&1
}
run "$OUT/lane0.log" $PY --pos newline --fit-docs 256 &
run "$OUT/lane1.log" $PY --pos newline --fit-docs 512 &
run "$OUT/lane2.log" $PY --pos timedigit --fit-docs 256 &
run "$OUT/lane3.log" $PY --pos timedigit --fit-docs 512 &
wait
grep -h "^layer\|^tracked\|^collected" "$OUT"/lane*.log
