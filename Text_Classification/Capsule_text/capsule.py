"""

@file  : capsule.py

@author: xiaolu

@time  : 2020-06-08

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

T_epsilon = 1e-7
num_classes = 2
Routings = 5
Dim_capsule = 16
gru_len = 128
Num_capsule = 10
dropout_p = 0.25


class Caps_Layer(nn.Module):
    def __init__(self, input_dim_capsule=gru_len * 2, num_capsule=Num_capsule, dim_capsule=Dim_capsule,
                 routings=Routings, share_weights=True, activation='default'):
        super(Caps_Layer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights

        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(torch.randn(Config.batch_size, input_dim_capsule, self.num_capsule * self.dim_capsule))

    def forward(self, x):
        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)   # (batch_size, max_len, hideen_size) x (1, hidden_size, 10x16)
        else:
            print('add later')
        batch_size = x.size(0)
        input_num_capsule = x.size(1)   # max_len
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        # 上一步相当于将hidden_size 分成 self.num_capsule x self.dim_capsule
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # 转成(batch_size, num_capsule, input_num_capsule, dim_capsule)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size, num_capsule, input_num_capsule)

        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))  # batch matrix multiplication
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication
        return outputs  # (batch_size, num_capsule, dim_capsule)

    def squash(self, x, axis=-1):
        # text version of squash, slight different from original one
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + T_epsilon)
        return x / scale


class Dense_Layer(nn.Module):
    '''
    全连接
    '''
    def __init__(self):
        super(Dense_Layer, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(Num_capsule * Dim_capsule, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)


class Capsule_Main(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None):
        super(Capsule_Main, self).__init__()
        self.embed_layer = nn.Embedding(Config.vocab_size, 300)
        self.gru = nn.GRU(input_size=300, hidden_size=gru_len,
                          bidirectional=True)
        self.caps_layer = Caps_Layer()
        self.dense_layer = Dense_Layer()

    def forward(self, content):
        content1 = self.embed_layer(content)
        # print(content1.size())   # torch.Size([2, 225, 300])
        content2, _ = self.gru(content1)
        # 这个输出是个tuple，一个output(batch_size, seq_len, num_directions * hidden_size)，一个hn
        # print(content2.size())   # torch.Size([2, 225, 256])

        content3 = self.caps_layer(content2)
        # print(content3.size())    # torch.Size([2, 10, 16])

        output = self.dense_layer(content3)
        # print(output.size())   # torch.Size([2, num_classes])

        return output