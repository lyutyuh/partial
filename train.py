# Code based on https://github.com/nikitakit/tetra-tagging/blob/master/examples/training.ipynb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from torch.nn.utils.rnn import pad_sequence

from tetratagger import BottomUpTetratagger

# assert torch.cuda.is_available()
device = torch.device("cuda")
print("Using device:", device)

READER = BracketParseCorpusReader('data', ['train', 'dev', 'test'])

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}


def ptb_unescape(sent):
    cleaned_words = []
    for word in sent:
        word = BERT_TOKEN_MAPPING.get(word, word)
        word = word.replace('\\/', '/').replace('\\*', '*')
        # Mid-token punctuation occurs in biomedical text
        word = word.replace('-LSB-', '[').replace('-RSB-', ']')
        word = word.replace('-LRB-', '(').replace('-RRB-', ')')
        if word == "n't" and cleaned_words:
            cleaned_words[-1] = cleaned_words[-1] + "n"
            word = "'t"
        cleaned_words.append(word)
    return cleaned_words


class TetraTaggingDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer, tag_system, pad_to_len=None,
                 max_train_len=60):
        assert split in ('train', 'dev', 'test')
        self.trees = READER.parsed_sents(split)
        self.tokenizer = tokenizer
        self.tag_system = tag_system
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_to_len = pad_to_len

        if split == 'train' and max_train_len is not None:
            # To speed up training, we only train on short sentences.
            self.trees = [
                tree for tree in self.trees if len(tree.leaves()) <= max_train_len]

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        tree = self.trees[index]
        words = ptb_unescape(tree.leaves())
        encoded = tokenizer.encode_plus(' '.join(words))
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        word_end_positions = [
            encoded.char_to_token(i)
            for i in np.cumsum([len(word) + 1 for word in words]) - 2]

        tag_ids = self.tag_system.tree_to_ids_pipeline(tree)

        # Pack both leaf and internal tag ids into a single "label" field.
        # (The huggingface API isn't flexible enough to use multiple label fields) 
        tag_ids = [tag_id + 1 for tag_id in tag_ids] + [0]
        tag_ids = torch.tensor(tag_ids, dtype=torch.long)
        labels = torch.zeros_like(input_ids)
        leaf_labels = tag_ids[::2] - self.tag_system.internal_tag_vocab_size
        internal_labels = tag_ids[1::2]
        labels[word_end_positions] = (
                internal_labels * (self.tag_system.leaf_tag_vocab_size + 1) + leaf_labels)

        if self.pad_to_len is not None:
            pad_amount = self.pad_to_len - input_ids.shape[0]
            assert pad_amount >= 0
            if pad_amount != 0:
                input_ids = F.pad(input_ids, [0, pad_amount], value=self.pad_token_id)
                labels = F.pad(labels, [0, pad_amount], value=0)

        return {'input_ids': input_ids, 'labels': labels}

    def collate(self, batch):
        input_ids = pad_sequence(
            [item['input_ids'] for item in batch],
            batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(
            [item['labels'] for item in batch],
            batch_first=True, padding_value=0)

        input_ids = input_ids.to(device)
        attention_mask = (input_ids != self.pad_token_id)
        labels = labels.to(device)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class ModelForTetraTagging(transformers.DistilBertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_leaf_labels = config.task_specific_params['num_leaf_labels']
        self.num_internal_labels = config.task_specific_params['num_internal_labels']

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if labels is not None:
            logits = outputs[0]
            internal_logits, leaf_logits = torch.split(
                logits, [self.num_internal_labels, self.num_leaf_labels], dim=-1)
            internal_labels = (labels // (self.num_leaf_labels + 1)) - 1
            leaf_labels = (labels % (self.num_leaf_labels + 1)) - 1

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_leaf_logits = leaf_logits.view(-1, self.num_leaf_labels)
                active_internal_logits = internal_logits.view(
                    -1, self.num_internal_labels)
                active_leaf_labels = torch.where(
                    active_loss, leaf_labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(leaf_labels)
                )
                active_internal_labels = torch.where(
                    active_loss, internal_labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(internal_labels)
                )
                loss = (loss_fct(active_leaf_logits, active_leaf_labels)
                        + loss_fct(active_internal_logits, active_internal_labels))
            else:
                loss = (loss_fct(leaf_logits.view(-1, self.num_leaf_labels),
                                 leaf_labels.view(-1))
                        + loss_fct(internal_logits.view(-1, self.num_internal_labels),
                                   internal_labels.view(-1)))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


print("Initialize Tag Vocabs")
tag_system = BottomUpTetratagger(trees=READER.parsed_sents('train'), add_remove_top=True)
print("Initialize Tokenizer")
tokenizer = transformers.AutoTokenizer.from_pretrained('distill-bert',
                                                       use_fast=True)
assert tokenizer.is_fast, "Only fast tokenizers are supported by this notebook"

print("Create Datasets")
train_dataset = TetraTaggingDataset('train', tokenizer, tag_system)

# The Trainer framework we're using doesn't allow variable-length sequences at
# evaluation time, so we pad to length 256.
eval_dataset = TetraTaggingDataset('dev', tokenizer, tag_system, pad_to_len=256)

print("Generate The Config")
config = transformers.AutoConfig.from_pretrained(
    'distill-bert',
    num_labels=len(tag_system.vocab_list),
    id2label={i: label for label, i in tag_system.vocab.items()},
    label2id={label: i for label, i in tag_system.vocab.items()},
    task_specific_params={
        'num_leaf_labels': tag_system.leaf_tag_vocab_size,
        'num_internal_labels': tag_system.internal_tag_vocab_size,
    }
)

print("Instantiate The Model")
model = ModelForTetraTagging.from_pretrained(
    'distill-bert', config=config)


def compute_metrics(p, num_leaf_labels=tag_system.leaf_tag_vocab_size):
    """Computes accuracies for both leaf and internal tagging decisions."""
    leaf_predictions = p.predictions[..., -num_leaf_labels:]
    internal_predictions = p.predictions[..., :-num_leaf_labels]
    leaf_labels = p.label_ids % (num_leaf_labels + 1) - 1
    internal_labels = p.label_ids // (num_leaf_labels + 1) - 1

    leaf_predictions = leaf_predictions[leaf_labels != -1].argmax(-1)
    internal_predictions = internal_predictions[internal_labels != -1].argmax(-1)

    leaf_labels = leaf_labels[leaf_labels != -1]
    internal_labels = internal_labels[internal_labels != -1]

    return {
        'internal_accuracy': (internal_predictions == internal_labels).mean(),
        'leaf_accuracy': (leaf_predictions == leaf_labels).mean(),
    }


print("Start Training!")
training_args = transformers.TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=160,
    weight_decay=0.01,
    logging_dir='./logs',
    save_steps=1149,
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=train_dataset.collate,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
