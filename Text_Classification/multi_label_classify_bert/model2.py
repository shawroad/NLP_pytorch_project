"""
@file   : model2.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-23
"""
import torch
from torch import nn
from config import set_args
from transformers.models.bert import BertConfig, BertModel


args = set_args()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        config = BertConfig.from_pretrained(args.bert_config, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(args.bert_pretrain, config=config)
        self.fc = nn.Linear(768, args.num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)  # [batch, seqlen, hidden_size]
        hidden_states = output[2][-4:]
        hidden = torch.stack(hidden_states, dim=-1).max(dim=-1)[0]  #[ batch, seqlen, hidden_size]
        output = self.fc(hidden[:, 0, :])
        output = torch.sigmoid(output)
        return output   # [batch, n_classes]