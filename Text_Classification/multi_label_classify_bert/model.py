"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-22
"""
import torch
from torch import nn
from config import set_args
from transformers.models.bert import BertModel, BertConfig


args = set_args()


class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained(args.bert_config)
        self.bert = BertModel.from_pretrained(args.bert_pretrain, config=self.config)
        self.fc = nn.Linear(self.bert.config.hidden_size, args.num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)[-1]
        output = self.fc(output)
        output = torch.sigmoid(output)
        return output
