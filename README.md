# Partial order dependency parser

This repository contains the code for partial order dependency parser.

## Setting Up The Environment
Set up a virtual environment and install the dependencies:
```bash
pip install -r requirements.txt
```

## Getting The Data
Follow the instructions in this [repo](https://github.com/nikitakit/self-attentive-parser/tree/master/data) to do the initial preprocessing on English WSJ and SPMRL datasets. The default data path is `data/spmrl` folder, where each file titled in `[LANGUAGE].[train/dev/test]` format.


## Building The Tagging Vocab
In order to use taggers, we need to build the vocabulary of tags for in-order, pre-order and post-order linearizations. You can cache these vocabularies using:
```bash
python run.py vocab --lang [LANGUAGE] --tagger [TAGGER]
```
Tagger can be `td-sr` for top-down (pre-order) shift--reduce linearization, `bu-sr` for bottom-up (post-order) shift--reduce linearization,
or `tetra` for in-order linearization.

## Training
Train the model and store the best checkpoint.
```bash
python run.py train --batch-size [BATCH_SIZE]  --tagger [TAGGER] --lang [LANGUAGE] --model [MODEL] --epochs [EPOCHS] --lr [LR] --model-path [MODEL_PATH] --output-path [PATH] --max-depth [DEPTH] --keep-per-depth [KPD] [--use-tensorboard]
```
- batch size: use 32 to reproduce the results
- tagger: `td-sr` or `bu-sr` or `tetra`
- lang: language, one of the nine languages reported in the paper
- model: `bert`, `bert+crf`, `bert+lstm`
- model path: path that pretrained model is saved
- output path: path to save the best trained model
- max depth: maximum depth to keep in the decoding lattice
- keep per depth: number of elements to keep track of in the decoding step
- use-tensorboard: whether to store the logs in tensorboard or not (true or false)

## Evaluation
Calculate evaluation metrics: fscore, precision, recall, loss.
```bash
python run.py evaluate --lang [LANGUAGE] --model-name [MODEL] --model-path [MODEL_PATH] --bert-model-path [BERT_PATH] --max-depth [DEPTH] --keep-per-depth [KPD]  [--is-greedy]
```
- lang: language, one of the nine languages reported in the paper
- model name: name of the checkpoint
- model path: path of the checkpoint
- bert model path: path to the pretrained model
- max depth: maximum depth to keep in the decoding lattice
- keep per depth: number of elements to keep track of in the decoding step
- is greedy: whether or not use the greedy decoding, default is false



#### PTB (English)
|    Model                 |    LAS       |     UAS      |
| ----------------------   | -----------  | -----------  |
| BERT-large               |    96.02     | 97.08        | bs = 64
| XLNet-large-cased        |    96.24     | 97.22        | # DEV: 95.93, 97.01
| RoBERTa                  |              |              | bs = 32 not as good as XLNet
| GPT2-large               |    72.07     | 68.11        | # uni-direction baseline


DEV: XLNet-large-cased: 

#### CTB (Chinese)
|    Model     |    LAS       |     UAS      |
| -----------  | -----------  | -----------  |
| BERT-base-chn       |    91.11  | 92.49        | # DEV: 90.15, 91.39 
| xlnet-mid-chn     |      91.91  | 93.19        | # DEV: 90.08, 91.40
| Biaffine-xlnet-mid-chn | 92.47  |   93.52  | # DEV: 91.69, 92.76


|    Model                 |    Speed  (sent/s)  (Avg. of 3 runs)    |
| ----------------------   | ---------------------  |
|  Hexatagger-BERT-large   |    3176.46             |  # 5306.59
|  biaffine-BERT-large     |    338.27              |


#### Hexatagger-BERT-large
|    Length     |    Speed  (sent/s)     | Memory (MB)   |
| -----------   | ---------------------  | ------------  |
|   32          |    2916.45     |   3011                |
|   64          |    3011.08     |   3123                |
|   128          |    2649.26     |  3763                |
|   256          |    3270.06     |  4659                |

#### Biaffine-BERT-large
|    Length     |    Speed  (sent/s)     | Memory (MB) (bs=128)  |
| -----------   | ---------------------  | ------------  |
|   32          |     493.73    |             4647 MB    |
|   64          |     328.63    |             10375 MB   |
|   128         |     202.34    |             31313 MB   | 
|   256         |      98.25    |             57593 MB (bs=64)  |


### UD 2.2
|   Language  |     LAS   |     UAS       |  
| ----------- | ----------| -----------   |
|             |           |               |
| bg          |   92.87   |  96.11        | 94.20, 97.46
| ca          |   93.79   |  95.41        | 94.64, 96.00
| cs          |   92.07   |  95.05        | 93.83, 96.26
| de          |   85.18   |  89.81        | 86.65, 91.18
| en          |   90.79   |  93.06        | 92.51, 94.50
| es          |   92.90   |  94.70        | 94.16, 95.60
| fr          |   91.15   |  93.92        | 94.08, 96.14
| it          |   94.33   |  95.93        | # xlm-xlnet-large 95.59, 96.91
| nl          |   91.46   |  94.02        | 94.06, 96.08
| no          |   93.89   |  95.80        | 95.89, 97.17
| ro          |   86.54   |  92.66        | 89.03, 94.07
| ru          |   93.89   |  95.48        | 95.67, 96.84
| avg.        |   91.58   |  94.03        | 93.36


### Commands to train the best models

#### PTB
```bash
python run.py train --lang English --max-depth 10 --tagger hexa --model bert --epochs 50 --batch-size 64 --lr 3e-5 --model-path bert-large-cased --output-path ./checkpoints/ --use-tensorboard True

# The best one:
python run.py train --lang English --max-depth 10 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 3e-5 --model-path xlnet-large-cased --output-path ./checkpoints/ --use-tensorboard True

```


#### CTB
```bash
python run.py train --lang Chinese --max-depth 10 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 5e-5 --model-path bert-base-chinese --output-path ./checkpoints/ --use-tensorboard True

python run.py train --lang Chinese --max-depth 10 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 3e-5 --model-path hfl/chinese-roberta-wwm-ext-large --output-path ./checkpoints/ --use-tensorboard True
```

#### UD
```bash
python run.py evaluate --lang bg --max-depth 10 --tagger hexa --bert-model-path bert-base-multilingual-cased --model-name bg-hexa-bert-1e-05-50 --batch-size 32 --model-path ./checkpoints/                                    
```


### Commands to evaluate
#### PTB
```bash
python run.py evaluate --lang English --max-depth 10 --tagger hexa --bert-model-path xlnet-large-cased --model-name English-hexa-bert-3e-05-50 --batch-size 64 --model-path ./
```

#### CTB
```bash
python run.py evaluate --lang Chinese --max-depth 10 --tagger hexa --bert-model-path bert-base-chinese --model-name Chinese-hexa-bert-3e-05-50 --batch-size 64 --model-path ./checkpoints/
```
#### UD
```bash
python run.py evaluate --lang bg --max-depth 10 --tagger hexa --bert-model-path bert-base-multilingual-cased --model-name bg-hexa-bert-1e-05-50 --batch-size 64 --model-path ./checkpoints/
```