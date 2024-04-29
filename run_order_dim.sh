#!/bin/bash

# iterate over k from 2 to 10
# execute: CUDA_VISIBLE_DEVICES=0 python run.py train --lang English --tagger part --model bert --epochs 50 --batch-size 32 --lr 2e-5 --order-dim k --n-lstm-layers 0 --model-path bert-base-cased --output-path ./checkpoints/ --use-tensorboard True
for k in {2..10}
do
    CUDA_VISIBLE_DEVICES=0 python run.py train --lang English --tagger part --model bert --epochs 50 --batch-size 32 --lr 2e-5 --order-dim $k --n-lstm-layers 0 --model-path bert-base-cased --output-path ./checkpoints/ --use-tensorboard True
done
