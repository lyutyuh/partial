# Partial order dependency parser

This repository contains the code for partial order dependency parser.

## Setting Up The Environment
Set up a virtual environment and install the dependencies:
```bash
conda create -n environment_partial.yml
# activate the environment
conda activate partial
```



### On CSCS Alps (Clariden, GH200)
No conda there; use the pytorch uenv plus a venv (`requirements-alps.txt` lists the extras):
```bash
bash setup_alps.sh                     # venv, data/ptb/*.gold.conllu symlinks, encoders -> /capstor/store cache
sbatch run_ptb_sbatch.sh               # 4 single-GPU arms on one node: {xlnet-large-cased, bert-base-cased} x K in {2, 4}
ARMS="xlnet-large-cased:2" EPOCHS=1 sbatch run_ptb_sbatch.sh   # custom arms
```
Notes: the uenv's transformers 4.57 / torch 2.9.1 replace the pinned 4.40.1 / 2.3.0; `nltk` must stay `<3.10`
(3.10 rejects corpus paths outside its sandbox). Evaluation during training uses the PTB **test** split.
The CTB/UD files are not wired in: `const.DEP_PATH` is fixed to `data/ptb/`.


## Getting The Data
PTB (Stanford dependencies 3.3.0) is included in `data/ptb/`; the loader expects `{train,dev,test}.gold.conllu`,
which `setup_alps.sh` creates as symlinks to the `ptb_*_3.3.0.sd.clean` files.



## Tests
CPU tests of the paper's claims (token-split structures, trees are 2-dimensional, Algorithm 1 vs. brute force,
the repo's scoring/loss with an exact realizer): `python -m pytest tests/ -q` (on Alps: prefix with
`uenv run --view=default pytorch/v2.9.1:v2 --` and use `.venv/bin/python`).


## Training

For running one single experiment:
```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --lang English --tagger part --model bert --epochs 50 --batch-size 32 --lr 2e-5 --order-dim 2 --n-lstm-layers 0 --model-path bert-base-cased --output-path ./checkpoints/ --use-tensorboard True
```
This command will train a partial order dependency parser on English PTB. 
Model weights will be saved in ./checkpoints/.

For running experiments in batch:
```bash
bash run_order_dim.sh
```

`run_order_dim.sh` looks like this:
```bash
#!/bin/bash

# running experiments for k from 2 to 10
for k in {2..10}
do
    CUDA_VISIBLE_DEVICES=0 python run.py train --lang English --tagger part --model bert --epochs 50 --batch-size 32 --lr 2e-5 --order-dim $k --n-lstm-layers 0 --model-path bert-base-cased --output-path ./checkpoints/ --use-tensorboard True
done
```


Submitting job using slurm:
```bash
sbatch --account=es_cott --ntasks=4 --time=24:00:00 --mem-per-cpu=4096 --gpus="rtx_4090:1" run_order_dim.sh;
```