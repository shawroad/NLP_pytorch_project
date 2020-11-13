# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/11/13 15:47:19
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import torch
from torch import nn
from transformers import XLNetConfig, XLNetModel
from transformers import XLNetTokenizer
from pdb import set_trace


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = XLNetConfig.from_pretrained('./xlnet_pretrain/config.json')
        self.xlnet = XLNetModel.from_pretrained('./xlnet_pretrain/pytorch_model.bin', config=self.config)
        self.fc = nn.Linear(self.config.d_model, 2)

    def forward(self, input_ids, attention_mask):
        output = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = torch.max(output, dim=1)[0]
        logits = self.fc(output)
        return logits