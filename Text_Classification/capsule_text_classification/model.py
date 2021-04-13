"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-04-13
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import set_args

args = set_args()


class Embed_Layer(nn.Module):
    def __init__(self, use_pre_embed, embedding_matrix=None, vocab_size=None, embedding_dim=300):
        super(Embed_Layer, self).__init__()
        self.embed = nn.Embedding(vocab_size + 1, embedding_dim)
        if use_pre_embed:
            self.embed.weight.data.copy_(torch.from_numpy(embedding_matrix))

    def forward(self, x, dropout_p=0.25):
        return nn.Dropout(p=dropout_p)((self.embed(x)))


class GRU_Layer(nn.Module):
    def __init__(self, gru_hidden_size):
        super(GRU_Layer, self).__init__()
        self.gru_hidden_size = gru_hidden_size
        self.gru = nn.GRU(input_size=300,
                          hidden_size=self.gru_hidden_size,
                          bidirectional=True)

    def init_weights(self):
        # 参数初始化
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


class Caps_Layer(nn.Module):
    def __init__(self, input_dim_capsule, num_capsule=10, dim_capsule=16, activation='default'):
        super(Caps_Layer, self).__init__()
        self.input_dim_capsule = input_dim_capsule
        self.batch_size = args.batch_size
        self.num_capsule = num_capsule   # 可以理解为self-attention中的多头
        self.dim_capsule = dim_capsule   # 每个头的维度
        self.share_weights = True
        self.routings = 5
        self.activation = activation
        self.T_epsilon = 1e-7

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, self.input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                torch.randn(self.batch_size, self.input_dim_capsule, self.num_capsule * self.dim_capsule))  # 64即batch_size

        if self.activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        u_hat_vecs = 0
        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)   # torch.Size([64, 100, 160])
        else:
            print('add later')
        # print(u_hat_vecs.size())   # torch.Size([64, 100, 160])
        batch_size = x.size(0)
        input_num_capsule = x.size(1)

        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        # print(u_hat_vecs.size())   # torch.Size([64, 100, 10, 16])

        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)   # 64, 10, 100, 16
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # 转置  torch.Size([64, 10, 100])

        outputs = 0
        for i in range(self.routings):
            b = b.permute(0, 2, 1)  # torch.Size([64, 100, 10])
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)  # torch.Size([64, 10, 100])
            b = b.permute(0, 2, 1)  # torch.Size([64, 10, 100])
            outputs = self.activation(torch.einsum('bij, bijk->bik', (c, u_hat_vecs)))
            if i < self.routings - 1:
                b = torch.einsum('bik, bijk->bij', (outputs, u_hat_vecs))
        return outputs

    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + self.T_epsilon)
        return x / scale


class Dense_Layer(nn.Module):
    def __init__(self, num_capsule, dim_capsule, num_classes):
        super(Dense_Layer, self).__init__()
        self.dropout_p = 0.25
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_classes = num_classes

        self.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_p, inplace=True),
            nn.Linear(self.num_capsule * self.dim_capsule, self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)


class Model(nn.Module):
    def __init__(self, use_pre_embed=False, embedding_matrix=None, vocab_size=None,
                 gru_hidden_size=128):
        super(Model, self).__init__()
        self.num_capsule = 10   # 可以理解为多头注意力中的头个数
        self.dim_capsule = 16   # 每个头的维度
        self.num_classes = 30   # 类别数
        # 1. 词嵌入
        self.embed_layer = Embed_Layer(use_pre_embed, embedding_matrix, vocab_size)

        # 2. 编码
        self.gru_layer = GRU_Layer(gru_hidden_size)
        self.gru_layer.init_weights()

        # 3. Capsule
        self.caps_layer = Caps_Layer(input_dim_capsule=2*gru_hidden_size,
                                     num_capsule=self.num_capsule,
                                     dim_capsule=self.dim_capsule)

        # 4. 输出
        self.dense_layer = Dense_Layer(num_capsule=self.num_capsule,
                                       dim_capsule=self.dim_capsule,
                                       num_classes=self.num_classes)

    def forward(self, content):
        content1 = self.embed_layer(content)
        # print(content1.size())   # torch.Size([64, 100, 300])

        content2, _ = self.gru_layer(content1)
        # 这个输出是个tuple，一个output(batch_size, seq_len, num_directions * hidden_size)，一个hn
        # print(content2.size())    # torch.Size([64, 100, 256])

        content3 = self.caps_layer(content2)
        # print(content3.size())   # torch.Size([64, 10, 16])

        output = self.dense_layer(content3)
        return output
