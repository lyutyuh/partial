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
sbatch run_ptb_sbatch.sh               # 4 single-GPU arms: xlnet-large-cased, K = 2, {Triton linear, quadratic} x seeds {1, 2}
ARMS="xlnet-large-cased:2:linear:1" EPOCHS=1 sbatch run_ptb_sbatch.sh   # arm = encoder:K:linear|quadratic:seed
```
Notes: the uenv's transformers 4.57 / torch 2.9.1 replace the pinned 4.40.1 / 2.3.0; `nltk` must stay `<3.10`
(3.10 rejects corpus paths outside its sandbox). Checkpoints are selected on PTB dev; the summary json also records
test LAS/UAS at the best dev epoch. `--linear-time` trains and decodes with the Triton kernels below.
The CTB/UD files are not wired in: `const.DEP_PATH` is fixed to `data/ptb/`.


## Getting The Data
PTB (Stanford dependencies 3.3.0) is included in `data/ptb/`; the loader expects `{train,dev,test}.gold.conllu`,
which `setup_alps.sh` creates as symlinks to the `ptb_*_3.3.0.sd.clean` files.



## Tests
CPU tests of the paper's claims (token-split structures, trees are 2-dimensional, Algorithm 1 vs. brute force,
the repo's scoring/loss with an exact realizer): `python -m pytest tests/ -q` (on Alps: prefix with
`uenv run --view=default pytorch/v2.9.1:v2 --` and use `.venv/bin/python`).


## Linear-time Triton kernels
`learning/linear_order.py` implements Alg. 1 (K = 2, hard max of Eq. 2) without the (N, N) score matrix:
`log_partition(f, g, lengths)` (the per-word arc softmax normaliser, with a Triton backward), `decode(...)`
(greedy heads), and `arc_loss(f, g, heads, lengths)` (equal to the repo's arc cross-entropy with hard max). `f`, `g` are
the two `(B, N, 2)` halves of the realizer output (`tosets`, `tosets_prime`). One program per sentence: bitonic sort of
heads and dependents by f1 - f2, then forward/reverse log-space scans. The model itself still uses the quadratic,
smooth-max scores unless `run.py train --linear-time` is given.
Tests: `python -m pytest tests/test_linear_order.py -q` (GPU, or the Triton interpreter with `CUDA_VISIBLE_DEVICES=`).
Benchmark: `python scripts/bench_linear_order.py` (GH200, batch 32, fwd+bwd of Z): flat ~0.65 ms for N = 32..4096 vs
0.6 / 3.6 / 13.7 / 54.9 ms for the score matrix at N = 256 / 1024 / 2048 / 4096 (memory 84 MB -> 21.5 GB vs < 10 MB).

## General-K aggregation (K >= 3)
`learning/order_k.py` extends the linear-time idea to any order dimension: branch k of Eq. 2 is a (K-1)-dimensional
dominance sum, answered by a range tree on dyadic blocks built from batched `sort` / `searchsorted` / `logcumsumexp`
(`cummax` for decoding). Exact for any K, differentiable through autograd, O(N log^(K-1) N) time and
O(N log^(K-2) N) memory; all K branches and all levels of a depth are fused into one sort. `log_partition_k`,
`decode_k`, `arc_loss_k` mirror the K = 2 API. Tests: `python -m pytest tests/test_order_k.py -q`; benchmark:
`python scripts/bench_order_k.py`. Measured on a GH200 (batch 32, Z fwd+bwd): K = 3 crosses the quadratic path at
N ~ 2k (2.1x faster at 4k, 40x less memory); K = 4 stays slower than the matrix up to N = 16k (work-bound, large
constants) and K = 5 is impractical. For K = 2 use the Triton kernel (`linear_order.py`), 3-4x faster than this path.
A hand-written backward (`autograd=False`: one transposed dominance pass, upstream gradient split by sign) is exact
but not faster and peaks at twice the memory; the peak is the forward's flattened tree (all level tuples at once).

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