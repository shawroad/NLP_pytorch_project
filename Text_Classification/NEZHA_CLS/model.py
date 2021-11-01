"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-09-30
"""
import torch
from torch import nn
from NEZHA.configuration_nezha import NeZhaConfig
from NEZHA.modeling_nezha import NeZhaModel


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 1. 实例化模型
        self.config = NeZhaConfig.from_pretrained('./nezha_pretrain')
        self.nezha = NeZhaModel.from_pretrained('./nezha_pretrain')

        self.dropout = nn.Dropout(p=0.1)
        self.n_classes = 6
        self.classifier = nn.Linear(
            self.config.hidden_size * 2, self.n_classes)
        self.bilstm = nn.LSTM(input_size=self.config.hidden_size,
                              hidden_size=self.config.hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input_ids, attention_mask):
        output = self.nezha(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = output[0]
        pooler_output = output[1]
        # hidden_states = output[2]
        seq_avg = torch.mean(sequence_output, dim=1)
        concat_out = torch.cat((seq_avg, pooler_output), dim=1)
        concat_out = self.dropout(concat_out)
        logit = self.classifier(concat_out)
        return logit