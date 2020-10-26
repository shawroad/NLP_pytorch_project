# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/10/21 19:48:07
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
from torch import nn
from transformers import BertModel, BertConfig


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./wobert_pretrain/bert_config.json')
        self.wobert = BertModel.from_pretrained('./wobert_pretrain/pytorch_model.bin', config=self.config)
        self.output = nn.Linear(self.config.hidden_size, 1)
    
    def forward(self, input_ids=None, attention_mask=None, segment_ids=None, labels=None):
        sequence_output, cls_output = self.wobert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        logits = self.output(cls_output)    # size: (batch_size, label_nums)
        return logits





        
