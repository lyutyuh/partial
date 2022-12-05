import numpy as np
import json
import os

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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


class TaggingDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer, tag_system, reader, device, is_tetratags=False, pad_to_len=None,
                 max_train_len=None):
        self.reader = reader
        self.trees = self.reader.parsed_sents(split)
        self.tokenizer = tokenizer
        self.tag_system = tag_system
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pad_to_len = pad_to_len
        self.device = device
        self.is_tetratags = is_tetratags

        if split == 'train' and max_train_len is not None:
            # To speed up training, we only train on short sentences.
            self.trees = [
                tree for tree in self.trees if len(tree.leaves()) <= max_train_len]
        if not os.path.exists("./pos.json") and "train" in split:
            self.pos_dict = self.get_pos_dict()
            with open("./pos.json", 'w') as fp:
                json.dump(self.pos_dict, fp)
        else:
            with open("./pos.json", 'r') as fp:
                self.pos_dict = json.load(fp)

    def get_pos_dict(self):
        pos_dict = {}
        for t in self.trees:
            for _, x in t.pos():
                pos_dict[x] = pos_dict.get(x, 1+len(pos_dict))
        return pos_dict

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        tree = self.trees[index]
        words = ptb_unescape(tree.leaves())
        pos_tags = [self.pos_dict.get(x[1], 0) for x in tree.pos()]

        if 'albert' in str(type(self.tokenizer)):
            # albert is case insensitive!
            words = [w.lower() for w in words]
        encoded = self.tokenizer.encode_plus(' '.join(words))
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
        pos_ids = torch.zeros_like(input_ids).to(self.device)


        word_end_positions = [
            encoded.char_to_token(i)
            for i in np.cumsum([len(word) + 1 for word in words]) - 2]

        tag_ids = self.tag_system.tree_to_ids_pipeline(tree)

        # Pack both leaf and internal tag ids into a single "label" field.
        # (The huggingface API isn't flexible enough to use multiple label fields)
        tag_ids = [tag_id + 1 for tag_id in tag_ids] + [0]
        tag_ids = torch.tensor(tag_ids, dtype=torch.long).to(self.device)
        labels = torch.zeros_like(input_ids).to(self.device)

        odd_labels = tag_ids[1::2]
        if self.is_tetratags:
            even_labels = tag_ids[::2] - self.tag_system.decode_moderator.internal_tag_vocab_size
            labels[word_end_positions] = (
                    odd_labels * (self.tag_system.decode_moderator.leaf_tag_vocab_size + 1) + even_labels)
            pos_ids[word_end_positions] = torch.tensor(pos_tags, dtype=torch.long).to(self.device)
        else:
            even_labels = tag_ids[::2]
            labels[word_end_positions] = (
                odd_labels * (len(self.tag_system.tag_vocab) + 1) + even_labels)

        if self.pad_to_len is not None:
            pad_amount = self.pad_to_len - input_ids.shape[0]
            assert pad_amount >= 0
            if pad_amount != 0:
                input_ids = F.pad(input_ids, [0, pad_amount], value=self.pad_token_id)
                pos_ids = F.pad(pos_ids, [0, pad_amount], value=0)
                labels = F.pad(labels, [0, pad_amount], value=0)

        return {
            'input_ids': input_ids,
            'pos_ids': pos_ids,
            'labels': labels
        }

    def collate(self, batch):
        input_ids = pad_sequence(
            [item['input_ids'] for item in batch],
            batch_first=True, padding_value=self.pad_token_id)
        pos_ids = pad_sequence(
            [item['pos_ids'] for item in batch],
            batch_first=True, padding_value=0) # 0 for UNK POS
        labels = pad_sequence(
            [item['labels'] for item in batch],
            batch_first=True, padding_value=0)

        input_ids = input_ids.to(self.device)
        attention_mask = (input_ids != self.pad_token_id).float()
        labels = labels.to(self.device)
        return {
            'input_ids': input_ids,
            'pos_ids': pos_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
