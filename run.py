import logging

import torch
import transformers
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from torch.utils.data import DataLoader
from transformers import AdamW

from learning.dataset import TaggingDataset
from learning.model import ModelForTetratagging, BertCRFModel, BertLSTMModel
from tagging.srtagger import SRTaggerBottomUp, SRTaggerTopDown
from tagging.tetratagger import BottomUpTetratagger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reader = BracketParseCorpusReader('data', ['train', 'dev', 'test'])

TETRATAGGER = "tetratagger"
TD_SR = "top-down shift-reduce"
BU_SR = "bottom-up shift-reduce"
MODEL_NAME = 'distilbert'


def initialize_tag_system(tagging_schema):
    if tagging_schema == BU_SR:
        tag_system = SRTaggerBottomUp(trees=reader.parsed_sents('train'))
    elif tagging_schema == TD_SR:
        tag_system = SRTaggerTopDown(trees=reader.parsed_sents('train'))
    elif tagging_schema == TETRATAGGER:
        tag_system = BottomUpTetratagger(trees=reader.parsed_sents('train'))
    else:
        logging.error("Please specify the tagging schema")
        return
    return tag_system


def prepare_training_data(tag_system, tagging_schema):
    is_tetratags = True if tagging_schema == TETRATAGGER else False
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, truncation=True,
                                                           use_fast=True)
    train_dataset = TaggingDataset('train', tokenizer, tag_system, reader, device,
                                   is_tetratags=is_tetratags)
    eval_dataset = TaggingDataset('dev', tokenizer, tag_system, reader, device,
                                  pad_to_len=256, is_tetratags=is_tetratags)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=16, collate_fn=train_dataset.collate
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=16, collate_fn=eval_dataset.collate
    )
    return train_dataset, eval_dataset, train_dataloader, eval_dataloader


def generate_config(model_type, tagging_shcema, tag_system):
    if model_type == "bert+crf" or model_type == "bert":
        config = transformers.AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=2 * len(tag_system.tag_vocab),
            task_specific_params={
                'num_tags': len(tag_system.tag_vocab),
            }
        )
    elif model_type == "bert" and tagging_shcema == TETRATAGGER:
        config = transformers.AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=len(tag_system.tag_vocab),
            id2label={i: label for i, label in enumerate(tag_system.tag_vocab)},
            label2id={label: i for i, label in enumerate(tag_system.tag_vocab)},
            task_specific_params={
                'num_even_tags': tag_system.leaf_tag_vocab_size,
                'num_odd_tags': tag_system.internal_tag_vocab_size,
            }
        )
    elif model_type == "bert" and tagging_shcema != TETRATAGGER:
        config = transformers.AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=2 * len(tag_system.tag_vocab),
            task_specific_params={
                'num_even_tags': len(tag_system.tag_vocab),
                'num_odd_tags': len(tag_system.tag_vocab),
            }
        )
    else:
        logging.error("Invalid combination of model type and tagging schema")
        return
    return config


def initialize_model(model_type, tagging_schema, tag_system):
    config = generate_config(model_type, tagging_schema, tag_system)
    if model_type == "bert+crf":
        model = BertCRFModel(config=config)
    elif model_type == "bert+lstm":
        model = BertLSTMModel(config=config)
    elif model_type == "bert":
        model = ModelForTetratagging.from_pretrained(MODEL_NAME, config=config)
    else:
        logging.error("Invalid model type")
        return
    return model


def initialize_optimizer_and_scheduler(model, train_dataloader, lr=5e-5, num_epochs=4,
                                       num_warmup_steps=160):
    optimizer = AdamW(model.parameters(), lr=lr)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=160,  # 160
        num_training_steps=num_training_steps,
    )
    return optimizer, lr_scheduler, num_warmup_steps

def initialized_training_pipeline(args):
    pass

