import torch
import transformers
from torch import nn
from transformers import DistilBertForTokenClassification

from learning.crf import CRF


class ModelForConditionalIndependance(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_tags = config.task_specific_params['num_tags']
        self.bert = DistilBertForTokenClassification.from_pretrained('distill-bert',
                                                                     config=config)
        self.crf = CRF(
            self.num_tags,
            batch_first=True,
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
            odd_labels = (labels // (self.num_tags + 1)) - 1
            even_labels = (labels % (self.num_tags + 1)) - 1

            extended_labels = torch.zeros((batch_size, seq_length * 2), dtype=torch.long).to(
                device)
            extended_labels[:, ::2] = even_labels
            extended_labels[:, 1::2] = odd_labels

            mask[extended_labels == -1] = 0
            extended_labels[extended_labels == -1] = 0

            nll = self.crf(em, extended_labels, mask=mask)
        return nll, logits


class ModelForTetraTagging(transformers.DistilBertForTokenClassification):
    # From tetratagging paper
    def __init__(self, config):
        super().__init__(config)
        self.num_tags = config.task_specific_params['num_tags']

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

            odd_logits, even_logits = torch.split(
                logits, [self.num_tags, self.num_tags], dim=-1)
            odd_labels = (labels // (self.num_tags + 1)) - 1
            even_labels = (labels % (self.num_tags + 1)) - 1

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_even_logits = even_logits.view(-1, self.num_tags)
                active_odd_logits = odd_logits.view(
                    -1, self.num_tags)
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
                loss = (loss_fct(even_logits.view(-1, self.num_tags),
                                 even_labels.view(-1))
                        + loss_fct(odd_logits.view(-1, self.num_tags),
                                   odd_labels.view(-1)))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
