"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-30$
"""
import torch
from torch import nn
from config import set_args
from pdb import set_trace
from transformers.models.bert import BertPreTrainedModel, BertModel

args = set_args()


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if args.pool_mode == "concat":
            self.classifier = nn.Linear(config.hidden_size * args.num_pool_layers, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        # outputs[0].size: torch.Size([12, 512, 768])
        # outputs[1].size: torch.Size([12, 768])
        # len(outputs[2]): 13

        if args.pool_mode == "concat":
            # 最后若干层cls向量的拼接
            selected_outputs = []
            for i in range(-args.num_pool_layers, 0):
                selected_outputs.append(outputs[2][i][:, 0, :])
            pooled_output = torch.cat(selected_outputs, dim=1)

        elif args.pool_mode == "mean":
            # 最后若干层进行平均池化
            selected_outputs = []
            for i in range(-args.num_pool_layers, 0):
                selected_outputs.append(outputs[2][i][:, 0, :].unsqueeze(1))
            selected_outputs = torch.cat(selected_outputs, dim=1)
            pooled_output = torch.mean(selected_outputs, dim=1)

        elif args.pool_mode == "max":
            # 最后若干层进行最大化池化
            selected_outputs = []
            for i in range(-args.num_pool_layers, 0):
                selected_outputs.append(outputs[2][i][:, 0, :].unsqueeze(1))
            selected_outputs = torch.cat(selected_outputs, dim=1)
            pooled_output, _ = torch.max(selected_outputs, dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
