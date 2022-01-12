import argparse
import logging

import torch
import transformers
import wandb
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm as tq

from learning.dataset import TaggingDataset
from learning.model import ModelForTetratagging, BertCRFModel, BertLSTMModel
from tagging.srtagger import SRTaggerBottomUp, SRTaggerTopDown
from tagging.tetratagger import BottomUpTetratagger
from learning.evaluate import predict, calc_parse_eval, calc_tag_accuracy

logging.getLogger().setLevel(logging.INFO)

logging.getLogger().setLevel(logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reader = BracketParseCorpusReader('data', ['train', 'dev', 'test'])

TETRATAGGER = "tetra"
TD_SR = "td-sr"
BU_SR = "bu-sr"
BERT = "bert"
BERTCRF = "bert+crf"
BERTLSTM = "bert+lstm"

MODEL_NAME = "distilbert"

# project and entity names for wandb
# TODO: remove the credentials later
PROJECT = "pat"
ENTITY = "afraamini"

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')
train = subparser.add_parser('train')
evaluate = subparser.add_parser('evaluate')

train.add_argument('--tagger', choices=[TETRATAGGER, TD_SR, BU_SR], required=True,
                   help="Tagging schema")
train.add_argument('--model', choices=[BERT, BERTCRF, BERTLSTM], required=True,
                   help="Model architecture")

train.add_argument('--model-path', type=str, default='distilbert',
                   help="Bert model path or name")
train.add_argument('--output-path', type=str, default='pat-models/',
                   help="Path to save trained models")
train.add_argument('--use-wandb', type=bool, default=True,
                   help="Whether to use the wandb for logging the results make sure to add credentials to run.py if set to true")

train.add_argument('--lr', type=float, default=5e-5)
train.add_argument('--epochs', type=int, default=4)
train.add_argument('--batch-size', type=int, default=16)
train.add_argument('--num-warmup-steps', type=int, default=160)

evaluate.add_argument('--model-name', type=str, required=True)
evaluate.add_argument('--model-path', type=str, default='pat-models/')
evaluate.add_argument('--output-path', type=str, default='results/')
evaluate.add_argument('--batch-size', type=int, default=16)
evaluate.add_argument('--use-wandb', type=bool, default=True,
                   help="Whether to use the wandb for logging the results make sure to add credentials to run.py if set to true")


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


def prepare_training_data(tag_system, tagging_schema, batch_size):
    is_tetratags = True if tagging_schema == TETRATAGGER else False
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, truncation=True,
                                                           use_fast=True)
    train_dataset = TaggingDataset('train', tokenizer, tag_system, reader, device,
                                   is_tetratags=is_tetratags)
    eval_dataset = TaggingDataset('dev', tokenizer, tag_system, reader, device,
                                  pad_to_len=256, is_tetratags=is_tetratags)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, collate_fn=train_dataset.collate
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=eval_dataset.collate
    )
    return train_dataset, eval_dataset, train_dataloader, eval_dataloader


def generate_config(model_type, tagging_schema, tag_system, model_path):
    if model_type == BERTCRF or model_type == BERTLSTM:
        config = transformers.AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=2 * len(tag_system.tag_vocab),
            task_specific_params={
                'model_path': model_path,
                'num_tags': len(tag_system.tag_vocab),
            }
        )
    elif model_type == BERT and tagging_schema == TETRATAGGER:
        config = transformers.AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=len(tag_system.tag_vocab),
            id2label={i: label for i, label in enumerate(tag_system.tag_vocab)},
            label2id={label: i for i, label in enumerate(tag_system.tag_vocab)},
            task_specific_params={
                'model_path': model_path,
                'num_even_tags': tag_system.leaf_tag_vocab_size,
                'num_odd_tags': tag_system.internal_tag_vocab_size,
            }
        )
    elif model_type == BERT and tagging_schema != TETRATAGGER:
        config = transformers.AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=2 * len(tag_system.tag_vocab),
            task_specific_params={
                'model_path': model_path,
                'num_even_tags': len(tag_system.tag_vocab),
                'num_odd_tags': len(tag_system.tag_vocab),
            }
        )
    else:
        logging.error("Invalid combination of model type and tagging schema")
        return
    return config


def initialize_model(model_type, tagging_schema, tag_system, model_path):
    config = generate_config(model_type, tagging_schema, tag_system, model_path)
    if model_type == BERTCRF:
        model = BertCRFModel(config=config)
    elif model_type == BERTLSTM:
        model = BertLSTMModel(config=config)
    elif model_type == BERT:
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
    return optimizer, lr_scheduler, num_training_steps


def initialize_wandb(project_name, entity, run_name, args):
    wandb.init(project=project_name, entity=entity)
    wandb.run.name = run_name
    wandb.config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }
    return run_name


def train(args):
    logging.info("Initializing Tag System")
    tag_system = initialize_tag_system(args.tagger)
    logging.info("Preparing Data")
    train_dataset, eval_dataset, train_dataloader, eval_dataloader = prepare_training_data(
        tag_system, args.tagger, args.batch_size)
    logging.info("Initializing The Model")
    model = initialize_model(args.model, args.tagger, tag_system, args.model_path)
    optimizer, lr_scheduler, num_training_steps = initialize_optimizer_and_scheduler(model,
                                                                                     train_dataloader,
                                                                                     args.lr,
                                                                                     args.epochs,
                                                                                     args.num_warmup_steps)
    run_name = args.tagger + "-" + args.model + "-" + str(args.lr) + "-" + str(args.epochs)
    if args.use_wandb:
        run_name = initialize_wandb(PROJECT, ENTITY,
                                run_name, args)
    model.to(device)
    logging.info("Starting The Training Loop")
    model.train()
    for _ in tq(range(args.epochs)):
        for batch in tq(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            if args.use_wandb:
                wandb.log({"loss": loss})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    torch.save(model, args.output_path + run_name)


def get_tagging_schema_from_model_name(model_name):
    tagging_schema = model_name.split("-")[0]
    if tagging_schema == "td" or tagging_schema == "bu":
        return tagging_schema + "-sr"
    else:
        return tagging_schema


def evaluate(args):
    if args.use_wandb:
        wandb.init(project=PROJECT, entity=ENTITY, resume=True)
    tagging_schema = get_tagging_schema_from_model_name(args.model_name)
    logging.info("Initializing Tag System")
    tag_system = initialize_tag_system(tagging_schema)
    logging.info("Preparing Data")
    _, eval_dataset, _, eval_dataloader = prepare_training_data(
        tag_system, tagging_schema, args.batch_size)
    model = torch.load(args.model_path + args.model_name)
    if tagging_schema == TETRATAGGER:
        num_leaf_labels = tag_system.leaf_tag_vocab_size
        num_tags = len(tag_system.tag_vocab)
    else:
        num_leaf_labels = len(tag_system.tag_vocab)
        num_tags = 2*len(tag_system.tag_vocab)
    predictions, eval_labels = predict(model, eval_dataloader, len(eval_dataset),
                                       num_tags, device)
    calc_tag_accuracy(predictions, eval_labels, num_leaf_labels, args.use_wandb)
    calc_parse_eval(predictions, eval_labels, eval_dataset, tag_system, args.output_path,
                    args.model_name)


def main():
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)


if __name__ == '__main__':
    main()
