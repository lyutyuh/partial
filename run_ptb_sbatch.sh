#!/bin/bash
#SBATCH --job-name=partial_ptb
#SBATCH --partition=normal
#SBATCH --account=a0087
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=780G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/store/cscs/swissai/a0087/tianyu/partial/logs/partial_ptb_%j.log

# Partial-order dependency parser on English PTB (Liu et al., EMNLP 2023, Tab. 1).
# Packs 4 single-GPU arms on one GH200 node; each arm is "<hf-encoder>:<order-dim K>".
# Override with: ARMS="xlnet-large-cased:2 xlnet-large-cased:4" EPOCHS=50 sbatch run_ptb_sbatch.sh
# Encoders must already be in $HF_HUB_CACHE (compute nodes run with HF_HUB_OFFLINE=1); see README.
set -euo pipefail
ROOT=/capstor/store/cscs/swissai/a0087/tianyu/partial
REPO=/users/tliu/repos/partial
ARMS=${ARMS:-"xlnet-large-cased:2 xlnet-large-cased:4 bert-base-cased:2 bert-base-cased:4"}
EPOCHS=${EPOCHS:-50}
export HF_HOME=$ROOT/hf_cache HF_HUB_CACHE=$ROOT/hf_cache/hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$REPO"

for arm in $ARMS; do
  model=${arm%%:*}; k=${arm##*:}
  out=$ROOT/checkpoints/${model}-k${k}/   # run_name omits the encoder, so arms need separate dirs
  mkdir -p "$out"
  srun --exclusive --ntasks=1 --gpus=1 --cpus-per-task=72 --mem=190G \
       --uenv=pytorch/v2.9.1:v2 --view=default \
       bash -c "ulimit -c 0; .venv/bin/python run.py train --lang English --tagger part --model bert \
         --epochs $EPOCHS --batch-size 32 --lr 2e-5 --order-dim $k --n-lstm-layers 0 \
         --model-path $model --output-path $out --use-tensorboard True" \
       > "$ROOT/logs/partial_ptb_${SLURM_JOB_ID}_${model}-k${k}.log" 2>&1 &
done
wait
