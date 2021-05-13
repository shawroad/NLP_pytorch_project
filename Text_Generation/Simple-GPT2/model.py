"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-11
"""
import torch.nn as nn
import torch
from config import set_args

# @ 指的是矩阵乘
args = set_args()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 用于创建字典参数，来对字典里的字进行编码
        self.vocab_embed = nn.Embedding(args.vocab_num, args.embed_dim)

        # 用于位置编码
        self.positon_embed = nn.Embedding(args.pos_num, args.embed_dim)

        # 用于句子编码
        self.type_embed = nn.Embedding(args.type_num, args.embed_dim)

        self.block = []
        for _ in range(args.block_num):
            self.block.append(Block(True))

        self.drop_layer = nn.Dropout(0.1)
        self.block_layer = nn.Sequential(*self.block)
        # 定义一个输出层，对6层block做输出
        self.output_layer = nn.Linear(args.embed_dim, args.vocab_num, bias=False)

    def forward(self, vocab, positon):
        vocab = self.vocab_embed(vocab)
        positon = self.positon_embed(positon)
        drop = self.drop_layer(vocab + positon)
        # print(drop.size())   # batch_size, max_len, hidden_size

        block = self.block_layer(drop)
        output = self.output_layer(block)
        return output


class Block(nn.Module):
    def __init__(self, isMask=False):
        super(Block, self).__init__()
        # 在通道方面做归一化
        self.layer_normal1 = nn.LayerNorm(args.embed_dim)
        self.attention_layer = Attention(isMask)
        self.layer_normal2 = nn.LayerNorm(args.embed_dim)
        # 扩大参数量
        self.output_layer = nn.Sequential(
            nn.Linear(args.embed_dim, args.head_num * args.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(args.head_num * args.embed_dim, args.embed_dim)
        )
        self.drop_layer = nn.Dropout(0.1)

    def forward(self, data):
        # batch_size, max_len, hidden_size
        normal1 = self.layer_normal1(data)
        attention = self.attention_layer(normal1)
        attention = attention + data
        normal2 = self.layer_normal2(attention)
        output = self.output_layer(normal2)
        output = self.drop_layer(output)
        output = output + normal2
        # print(output.size())   # torch.Size([2, 200, 60])
        return output


class Attention(nn.Module):
    def __init__(self, isMask=False):
        super(Attention, self).__init__()
        self.dk = (args.embed_dim // args.head_num) ** 0.5
        self.isMask = isMask
        # 把一个词复制三份，Q,K,V
        self.copy_layer = nn.Linear(args.embed_dim, args.embed_dim * 3)

        self.drop_layer = nn.Dropout(0.1)
        # 在计算完注意力后来个线性层来计算注意力
        self.output_layer = nn.Linear(args.embed_dim, args.embed_dim)
        if self.isMask:
            # 使用tril删除右上角的数据，并使用register_buffer保存mask状态，使它不在bp算法被当作参数更新，但可以被传入CUAD或CPU进行计算，保存时会被当作权重保存，但不参与训练
            self.register_buffer("mask", torch.tril(torch.ones(args.pos_num, args.pos_num)))

    def forward(self, data):
        # print(data.size())   # torch.Size([2, 200, 60])
        data = self.copy_layer(data)   # batch_size, max_len, 3 * hidden_size

        data = data.reshape(*data.shape[:-1], args.head_num, -1)   # batch_size, max_len, 12,

        data = data.transpose(-2, -3)   #
        # n,12,s,64*3-->n,12,s,64
        q, k, v = data.chunk(3, dim=-1)
        # n,12,s,64-->n,12,s,s
        w = (q @ (k.transpose(-1, -2))) / self.dk
        # print(w.size())  # torch.Size([1, 12, 200, 200])   batch_size num_heads max_len max_len
        # print(self.mask.size())   # torch.Size([200, 200])
        # print(self.mask)    # 下三角矩阵

        if self.isMask:
            mask = self.mask[0:w.size(-2), 0:w.size(-1)]
            # 保证右上角无限小，这样经过softmax以后的值为0
            w = w * mask - (1 - mask) * 1e5  # 如果不让捕捉的位置  这里将是负无穷大 负责就是计算的值 计算softmax 就变为0了

        w = torch.softmax(w, dim=-1)
        w = self.drop_layer(w)
        # print(w.size())    # torch.Size([1, 12, 200, 200])
        # print(v.size())    # torch.Size([1, 12, 200, 5])

        value = w @ v
        value = value.transpose(-2, -3)
        value = value.reshape(*value.shape[0:-2], -1)
        output = self.output_layer(value)
        output = self.drop_layer(output)
        return output
