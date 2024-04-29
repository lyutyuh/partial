# Partial order dependency parser

This repository contains the code for partial order dependency parser.

## Setting Up The Environment
Set up a virtual environment and install the dependencies:
```bash
conda create -n environment_partial.yml
# activate the environment
conda activate partial
```



## Getting The Data



## Training

For running one single experiment:
```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --lang English --tagger part --model bert --epochs 50 --batch-size 32 --lr 2e-5 --order-dim 2 --n-lstm-layers 0 --model-path bert-base-cased --output-path ./checkpoints/ --use-tensorboard True
```
This command will train a partial order dependency parser on English PTB. 
Model weights will be saved in ./checkpoints/.


