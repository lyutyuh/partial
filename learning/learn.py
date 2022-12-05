import torch
from torch import nn

from transformers import AutoConfig, AutoModelForTokenClassification, AutoModel

from learning.crf import CRF


def calc_loss_helper(logits, labels, attention_mask, num_even_tags, num_odd_tags):
    odd_logits, even_logits = torch.split(
        logits, [num_odd_tags, num_even_tags], dim=-1)
    odd_labels = (labels // (num_even_tags + 1)) - 1
    even_labels = (labels % (num_even_tags + 1)) - 1

    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
    # Only keep active parts of the loss
    if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_even_logits = even_logits.view(-1, num_even_tags)
        active_odd_logits = odd_logits.view(
            -1, num_odd_tags)
        active_even_labels = torch.where(
            active_loss, even_labels.view(-1),
            torch.tensor(loss_fct.ignore_index).type_as(even_labels)
        )
        active_odd_labels = torch.where(
            active_loss, odd_labels.view(-1),
            torch.tensor(loss_fct.ignore_index).type_as(odd_labels)
        )
        loss = (loss_fct(active_even_logits, active_even_labels)
                + loss_fct(active_odd_logits, active_odd_labels))
    else:
        loss = (loss_fct(even_logits.view(-1, num_even_tags),
                         even_labels.view(-1))
                + loss_fct(odd_logits.view(-1, num_odd_tags),
                           odd_labels.view(-1)))
    return loss


class BertCRFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_tags = config.task_specific_params['num_tags']
        self.model_path = config.task_specific_params['model_path']
        is_eng = config.task_specific_params['is_eng']
        if is_eng:
            self.bert = AutoModelForTokenClassification.from_pretrained(
                self.model_path, config=config)
        else:
            self.bert = AutoModelForTokenClassification.from_pretrained(self.model_path,
                                                                         config=config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.crf = CRF(
            self.num_tags,
            batch_first=True,
            device=device
        )

    def forward(self, input_ids,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None, ):
        outputs = self.bert.forward(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, )

        logits = outputs[0]

        batch_size, seq_length, nb_tags = logits.shape
        em = logits.reshape(batch_size, seq_length * 2, -1)

        mask = attention_mask.repeat_interleave(2, dim=1).type(torch.uint8)

        nll = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if labels is not None:
            even_labels = (labels // (self.num_tags + 1)) - 1
            odd_labels = (labels % (self.num_tags + 1)) - 1

            extended_labels = torch.zeros((batch_size, seq_length * 2), dtype=torch.long).to(
                device)
            extended_labels[:, 1::2] = odd_labels
            extended_labels[:, ::2] = even_labels

            mask[extended_labels == -1] = 0
            extended_labels[extended_labels == -1] = 0

            nll = self.crf(em, extended_labels, mask=mask)
        return nll, logits


class BertLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_tags = config.task_specific_params['num_tags']
        self.model_path = config.task_specific_params['model_path']
        is_eng = config.task_specific_params['is_eng']
        if is_eng:
            self.bert = AutoModelForTokenClassification.from_pretrained(self.model_path,
                                                                     config=config)
        else:
            self.bert = AutoModelForTokenClassification.from_pretrained(self.model_path,
                                                                         config=config)
        self.lstm = nn.LSTM(
            self.num_tags, self.num_tags, 2, batch_first=True, bidirectional=True,
        )
        self.hidden2tag = nn.Linear(2 * self.num_tags, self.num_tags)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None, ):
        outputs = self.bert.forward(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, )

        logits = outputs[0]
        batch_size, seq_length, nb_tags = logits.shape
        em = logits.reshape(batch_size, seq_length * 2, -1)

        lstm_out = self.lstm(em)
        lstm_out = self.hidden2tag(lstm_out[0])
        lstm_out = self.softmax(lstm_out)
        lstm_out = lstm_out.reshape(batch_size, seq_length, -1)

        loss = None
        if labels is not None:
            loss = calc_loss_helper(lstm_out, labels, attention_mask, self.num_tags,
                                    self.num_tags)
        return loss, lstm_out


class ModelForTetratagging(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_even_tags = config.task_specific_params['num_even_tags']
        self.num_odd_tags = config.task_specific_params['num_odd_tags']
        self.model_path = config.task_specific_params['model_path']
        is_eng = config.task_specific_params['is_eng']
        self.use_pos = config.task_specific_params.get('use_pos', False)
        if self.use_pos:
            self.pos_encoder = nn.Sequential(
                nn.Embedding(50, 256, padding_idx=0)
            )
        if is_eng:
            self.bert = AutoModel.from_pretrained(self.model_path,
                                                  config=config)
        else:
            self.bert = AutoModel.from_pretrained(self.model_path,
                                                  config=config)
        self.projection = nn.Sequential(
            nn.Dropout(p=(1. / 3.)), # default dr in: https://github.com/yzhangcs/parser/blob/main/supar/model.py
            nn.Linear(config.hidden_size+(256*self.use_pos), config.num_labels)
        )

    def forward(
            self,
            input_ids=None,
            pos_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if self.use_pos:
            pos_encodings = self.pos_encoder(pos_ids)
            token_repr = torch.cat([outputs[0], pos_encodings], dim=-1)
        else:
            token_repr = outputs[0]

        tag_logits = self.projection(
            token_repr
        )

        loss = None
        if labels is not None:
            loss = calc_loss_helper(tag_logits, labels, attention_mask, self.num_even_tags,
                                    self.num_odd_tags)
        return loss, tag_logits
