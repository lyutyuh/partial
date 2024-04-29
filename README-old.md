# Hexatagging

This repository contains code for training and evaluation of Hexatagging: Projective Dependency Parsing as Tagging.


## Setting Up The Environment Using Conda

Create a virtual environment named `hexa` and install the dependencies:
```bash
conda create -f environment_hexa.yml
```

Activate `hexa` environment:
```bash
conda activate hexa
```

## Getting The Data
1. Convert CoNLL to Binary Headed Trees:
```bash
python data/dep2bht.py
```
This will generate the phrase-structured BHT trees in the `data/bht` directory.

2. Generate the vocabulary
```bash
python run.py vocab --lang English --tagger hexa
python run.py vocab --lang Chinese --tagger hexa
```


## Evaluation

### PTB (English)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --lang English --max-depth 6 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 2e-5 --model-path xlnet-large-cased --output-path ./checkpoints/ --use-tensorboard True
```

### CTB (Chinese)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --lang Chinese --max-depth 6 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 2e-5 --model-path hfl/chinese-xlnet-mid --output-path ./checkpoints/ --use-tensorboard True
```