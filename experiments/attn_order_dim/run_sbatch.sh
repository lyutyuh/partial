#!/bin/bash
#SBATCH --job-name=attn_order_dim
#SBATCH --partition=debug
#SBATCH --account=a0087
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=780G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/store/cscs/swissai/a0087/tianyu/partial/logs/attn_order_dim_%j.log

# Order dimension of attention argmax graphs in Qwen3-0.6B-Base and Qwen3-1.7B-Base (WikiText-103).
# 1) extract both models in parallel (GPU 0, 1), skipped when the data already exists; 2) fit: 2 GPUs per model,
# even / odd layers. CONFIGS (space-separated scorer:K:masked, see fit.py) and TAG select a run, e.g. the
# K = 2 head-combination run:
#   TAG=combo CONFIGS="order:2:masked osum:2:masked ..." sbatch --time=01:20:00 run_sbatch.sh
# TAUS (score temperatures, each config fitted once per tau) and VAL_SEQS (training sequences held out to select the
# best tau / steps / lr in report.py) give every scorer a tuned logit scale; sweep the scorers you compare alike, e.g.
# the references after fit.py's dot scaling fix (results before it used an unscaled dot):
#   TAG=refs TAUS="0.5 1 2" VAL_SEQS=64 CONFIGS="dot:16:masked dot:32:masked dot:64:masked dot:128:masked \
#       rope:128:masked order:2:masked order:4:masked osum:16:masked" sbatch --time=02:00:00 run_sbatch.sh
set -euo pipefail
ROOT=/capstor/store/cscs/swissai/a0087/tianyu/partial
OUT=$ROOT/attn_order_dim
REPO=/users/tliu/repos/partial
STEPS=${STEPS:-1500}
CONFIGS=${CONFIGS:-}
TAG=${TAG:-base}
TAUS=${TAUS:-1}
VAL_SEQS=${VAL_SEQS:-0}
LAYERS=${LAYERS:-$(seq 0 27 | tr '\n' ' ')}  # split alternately across the 2 GPUs of each model
export HF_HOME=$ROOT/hf_cache HF_HUB_CACHE=$ROOT/hf_cache/hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$REPO"
mkdir -p "$OUT/results"
run() {  # run <log> <cmd...> as a 1-GPU step
  local log=$1; shift
  srun --exclusive --ntasks=1 --gpus=1 --cpus-per-task=72 --mem=190G --uenv=pytorch/v2.9.1:v2 --view=default \
       bash -c "ulimit -c 0; $*" > "$log" 2>&1
}

for m in 0.6B 1.7B; do
  [ -f "$OUT/data/qwen3-${m}/test/layer27.pt" ] && continue
  run "$OUT/extract_${m}_${SLURM_JOB_ID}.log" .venv/bin/python experiments/attn_order_dim/extract.py \
      --model Qwen/Qwen3-${m}-Base --out "$OUT/data/qwen3-${m}" &
done
wait
cfg=""; [ -n "$CONFIGS" ] && cfg="--configs $CONFIGS"

for m in 0.6B 1.7B; do
  for parity in 0 1; do
    layers=$(echo $LAYERS | tr ' ' '\n' | awk -v p=$parity 'NR % 2 == 1 - p' | tr '\n' ' ')
    run "$OUT/fit_${m}_p${parity}_${TAG}_${SLURM_JOB_ID}.log" .venv/bin/python experiments/attn_order_dim/fit.py \
        --data "$OUT/data/qwen3-${m}" --layers $layers --steps $STEPS --taus $TAUS --val-seqs $VAL_SEQS $cfg \
        --out "$OUT/results/qwen3-${m}_p${parity}_${TAG}_${SLURM_JOB_ID}.jsonl" &
  done
done
wait
