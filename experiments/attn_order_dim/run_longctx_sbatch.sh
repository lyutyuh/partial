#!/bin/bash
#SBATCH --job-name=attn_longctx
#SBATCH --partition=debug
#SBATCH --account=a0087
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=780G
#SBATCH --time=01:30:00
#SBATCH --output=/capstor/store/cscs/swissai/a0087/tianyu/partial/logs/attn_longctx_%j.log

# Long-context head replacement (longctx.py), Qwen3-0.6B/1.7B-Base, WikiText-103: students for layers 2, 8, 14, 22-27
# trained online at T_train = 2048 with relative position slopes, evaluated at N = 512 .. 32768 on the same test tokens.
#   GPU 0/2 (lane a, 0.6B/1.7B): train order1 + rope128 + rope128s -> consistency (order1) -> ppl base/order1/
#                                rope128/rope128s (+ order1 subsets at 2k/32k) -> recall@k of order1 at 512/2k/8k/32k
#                                (all windows). rope128s = dot-product control with the order students' per-query sink
#                                logit (the matched control); rope128 (no sink) is kept for reference.
#   GPU 1/3 (lane b, 0.6B/1.7B): train omix4 -> consistency (omix4) -> ppl omix4 (+ subsets) -> order_attention bench
# Then the parts are merged into <OUT>/qwen3-<size>.json (ppl and NLL gap to base at every N). Students already in OUT
# are reused (resume: OUT=<old dir>).
# Expected wall time from login-node GH200 timings (1500 steps, full evaluation measured on the 1.7B):
#   1.7B lane a ~38 min (train ~0.44 s/step = 11 min with rope128s, ppl ~22 min, recall 5 min); lane b ~32 min
#   (train 0.55 s/step = 14 min, ppl 16 min); 0.6B lanes ~22 / ~23 min. Whole job ~40 min; STEPS=3000 ~55 min.
#   Peak memory of the 1.7B at 32k: 14 GB for every scorer (the lm_head chunk dominates; order_attention adds
#   3.3-3.6 GB; rope128s keeps the memory-efficient SDPA kernel).
set -euo pipefail
ROOT=/capstor/store/cscs/swissai/a0087/tianyu/partial
OUT=${OUT:-$ROOT/attn_order_dim/longctx/${SLURM_JOB_ID}}
REPO=/users/tliu/repos/partial
STEPS=${STEPS:-1500}
export HF_HOME=$ROOT/hf_cache HF_HUB_CACHE=$ROOT/hf_cache/hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$REPO"
mkdir -p "$OUT"
PY="$REPO/.venv/bin/python $REPO/experiments/attn_order_dim/longctx.py --out $OUT --steps $STEPS"
run() {
  local log=$1; shift
  srun --exclusive --ntasks=1 --gpus=1 --cpus-per-task=72 --mem=190G --uenv=pytorch/v2.9.1:v2 --view=default \
       bash -c "ulimit -c 0; $*" > "$log" 2>&1
}
for m in 0.6B 1.7B; do
  run "$OUT/lane_a_${m}.log" $PY --model Qwen/Qwen3-${m}-Base --part lane_a \
      --stages train consistency ppl recall --scorers order1 rope128 rope128s --subset-scorers order1 &
  run "$OUT/lane_b_${m}.log" $PY --model Qwen/Qwen3-${m}-Base --part lane_b \
      --stages train consistency ppl bench --scorers omix4 --subset-scorers omix4 --no-base &
done
wait
for m in 0.6B 1.7B; do
  run "$OUT/merge_${m}.log" $PY --model Qwen/Qwen3-${m}-Base --merge
  cat "$OUT/merge_${m}.log"
done
