from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader

from learning.dataset import TaggingDataset
from learning.model import ModelForConditionalIndependance
from tagging.srtagger import SRTaggerBottomUp
import transformers
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

reader = BracketParseCorpusReader('data', ['train', 'dev', 'test'])
tag_system = SRTaggerBottomUp(trees=reader.parsed_sents('train'))
tokenizer = transformers.AutoTokenizer.from_pretrained(
    'distilbert-base-uncased', use_fast=True)
assert tokenizer.is_fast, "Only fast tokenizers are supported"

train_dataset = TaggingDataset('train', tokenizer, tag_system, reader, device)
eval_dataset = TaggingDataset('dev', tokenizer, tag_system, reader, device, pad_to_len=256)

config = transformers.AutoConfig.from_pretrained(
    'distill-bert',
    num_labels = 2*len(tag_system.tag_vocab),
    task_specific_params = {
      'num_tags': len(tag_system.tag_vocab),
    }
)

model = ModelForConditionalIndependance.from_pretrained('pat-models', config=config)

def compute_metrics(p, num_leaf_labels=len(tag_system.tag_vocab)):
    """Computes accuracies for both leaf and internal tagging decisions."""
    print(p)
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

p = trainer.predict(eval_dataset)
print(p.metrics)