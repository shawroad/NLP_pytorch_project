# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/11/12 17:00:22
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
from torch import nn
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = LongformerConfig.from_pretrained('./longformer_pretrain')
        self.config.attention_mode = 'sliding_chunks'
        self.longformer = Longformer.from_pretrained('./longformer_pretrain', config=self.config)
        self.output = nn.Linear(self.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        sequence_output, cls_output = self.longformer(input_ids, attention_mask=attention_mask)
        logits = self.output(cls_output)
        return logits

        