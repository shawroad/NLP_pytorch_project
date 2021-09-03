"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-09-03
"""
import torch
from torch import nn
from pdb import set_trace
import torch.nn.functional as F
from transformers.models.bert import BertModel, BertConfig


class Caps_Layer(nn.Module):
    def __init__(self, input_dim_capsule, num_capsule, dim_capsule, routings):
        '''
        :param input_dim_capsule: 输入的维度 即: hidden_size
        :param num_capsule: capsule的个数
        :param dim_capsule: 每个capsule的维度
        :param routings: 进行几次路由计算
        '''
        super(Caps_Layer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.activation = self.squash

        self.W = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule))
        )

    def forward(self, x):
        # x.size(): batch_size, max_len, hidden_size
        batch_size = x.size(0)
        input_num_capsule = x.size(1)  # max_len

        u_hat_vecs = torch.matmul(x, self.W)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))

        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # 变为(batch_size, num_capsule, input_num_capsule, dim_capsule)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size, num_capsule, input_num_capsule)
        # print(b.size())   # torch.Size([2, 10, 128])

        outputs = None
        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(torch.einsum('bij,bijk->bik', c, u_hat_vecs))  # batch matrix multiplication
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', outputs, u_hat_vecs)  # batch matrix multiplication
        return outputs  # (batch_size, num_capsule, dim_capsule)

    def squash(self, x, axis=-1):
        T_epsilon = 1e-7
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + T_epsilon)
        return x / scale


class GRU_Layer(nn.Module):
    def __init__(self, gru_len=128):
        super(GRU_Layer, self).__init__()
        self.gru = nn.GRU(input_size=768, hidden_size=gru_len, bidirectional=True)

    def init_weights(self):
        # 初始化参数
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.orthogonal_(k)
        for k in b:
            nn.init.constant_(k, 0)

    def forward(self, x):
        return self.gru(x)


class Classifier(nn.Module):
    def __init__(self, dropout, num_capsule, dim_capsule, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(num_capsule * dim_capsule, num_classes),  # num_capsule*dim_capsule -> num_classes
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_capsule = 10
        self.dim_capsule = 16
        self.gru_len = 128
        self.Routings = 5
        self.dropout_p = 0.25
        self.num_classes = 31
        self.config = BertConfig.from_pretrained('../bert_pretrain/config.json')
        self.model = BertModel.from_pretrained('../bert_pretrain/pytorch_model.bin', config=self.config)
        self.gru = GRU_Layer(gru_len=self.gru_len)
        self.capsule = Caps_Layer(input_dim_capsule=2*self.gru_len,
                                  num_capsule=self.num_capsule,
                                  dim_capsule=self.dim_capsule,
                                  routings=self.Routings)

        self.output = Classifier(dropout=self.dropout_p,
                                 num_capsule=self.num_capsule,
                                 dim_capsule=self.dim_capsule,
                                 num_classes=self.num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output[0]  # torch.Size([2, 128, 768])
        sequence_output = sequence_output.permute(1, 0, 2)
        gru_output = self.gru(sequence_output)
        # print(gru_output[0].size())    # torch.Size([128, 2, 256])
        # print(gru_output[1].size())    # torch.Size([2, 2, 128])
        gru_encode = gru_output[0].permute(1, 0, 2)
        caps_output = self.capsule(gru_encode)    # torch.Size([2, 10, 16])
        logits = self.output(caps_output)
        return logits
