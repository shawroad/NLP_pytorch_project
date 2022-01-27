"""
@file   : model_cls.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-01-05
"""
import torch
from torch import nn
from pdb import set_trace
from model.modeling_nezha import NeZhaModel, NeZhaConfig


class Highway(nn.Module):
    # 加个highway网络
    def __init__(self, size, num_layers):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = nn.ReLU()

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x


class Classifier(nn.Module):
    # 加个全连接 进行分类
    def __init__(self, num_cls):
        super(Classifier, self).__init__()
        self.dense1 = torch.nn.Linear(768, 384)
        self.dense2 = torch.nn.Linear(384, num_cls)
        self.activation = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = NeZhaConfig.from_pretrained('./nezha_pretrain/config.json')
        self.config.output_hidden_states = True   # 输出所有的隐层
        self.config.output_attentions = True  # 输出所有注意力层计算结果
        self.nezha = NeZhaModel.from_pretrained('./nezha_pretrain', config=self.config)

        num_cls = 21
        self.highway = Highway(size=768, num_layers=3)
        self.classifier = Classifier(num_cls)

    def forward(self, input_ids, attention_mask, segment_ids):
        output = self.nezha(input_ids=input_ids, attention_mask=attention_mask)
        # output[0].size(): batch_size, max_len, hidden_size
        # output[1].size(): batch_size, hidden_size
        # len(output[2]): 13, 其中一个元素的尺寸: batch_size, max_len, hidden_size
        # len(output[3]): 12, 其中一个元素的尺寸: batch_size, 12层, max_len, max_len

        cls_output = output[1]
        hw_output = self.highway(cls_output)
        logits = self.classifier(hw_output)
        return logits