"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-22
"""
import torch
from torch import nn
from roformer.modeling_roformer import RoFormerModel
from pdb import set_trace


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.roformer = RoFormerModel.from_pretrained('./roformer_pretrain')
        self.classifier = Classifier()

    def forward(self, ids, mask, token_type_ids):
        sequence_output = self.roformer(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)[0]
        # sequence_output 的 size: batch_size, max_len, hidden_size
        # 将CLS取出
        output_1 = sequence_output[:, 0, :]
        output = self.classifier(output_1)
        return output


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.dense1 = nn.Linear(768, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 31)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense3(x)
        return x
