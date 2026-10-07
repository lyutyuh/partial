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

# Partial-order dependency parser on English PTB (Liu et al., EMNLP 2023, Tab. 1; hyperparameters of App. F.1:
# 3-layer BiLSTM, batch 32, lr 2e-5, 50 epochs). Packs 4 single-GPU arms on one GH200 node.
# Arm = "<hf-encoder>:<order-dim K>:<linear|quadratic>:<seed>"; linear = Triton kernels (K = 2 only).
# Model selection is on dev; the summary json records test LAS/UAS at the best dev epoch.
# Override with: ARMS="xlnet-large-cased:2:linear:1" EPOCHS=1 sbatch run_ptb_sbatch.sh
# Encoders must already be in $HF_HUB_CACHE (compute nodes run with HF_HUB_OFFLINE=1); see README.
set -euo pipefail
ROOT=/capstor/store/cscs/swissai/a0087/tianyu/partial
REPO=/users/tliu/repos/partial
ARMS=${ARMS:-"xlnet-large-cased:2:linear:1 xlnet-large-cased:2:linear:2 xlnet-large-cased:2:quadratic:1 xlnet-large-cased:2:quadratic:2"}
EPOCHS=${EPOCHS:-50}
LSTM_LAYERS=${LSTM_LAYERS:-3}
export HF_HOME=$ROOT/hf_cache HF_HUB_CACHE=$ROOT/hf_cache/hub HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$REPO"

for arm in $ARMS; do
  IFS=: read -r model k mode seed <<< "$arm"
  name=${model}-k${k}-${mode}-s${seed}
  out=$ROOT/checkpoints/${SLURM_JOB_ID}/${name}/   # run_name omits encoder and seed, so arms need separate dirs
  mkdir -p "$out"
  flag=""; [ "$mode" = linear ] && flag="--linear-time"
  srun --exclusive --ntasks=1 --gpus=1 --cpus-per-task=72 --mem=190G \
       --uenv=pytorch/v2.9.1:v2 --view=default \
       bash -c "ulimit -c 0; .venv/bin/python run.py train --lang English --tagger part --model bert \
         --epochs $EPOCHS --batch-size 32 --lr 2e-5 --order-dim $k --n-lstm-layers $LSTM_LAYERS $flag \
         --seed $seed --model-path $model --output-path $out --use-tensorboard True" \
       > "$ROOT/logs/partial_ptb_${SLURM_JOB_ID}_${name}.log" 2>&1 &
done
wait
