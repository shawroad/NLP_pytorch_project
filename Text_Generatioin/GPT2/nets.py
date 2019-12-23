"""

@file  : nets.py

@author: xiaolu

@time  : 2019-11-20

"""
import torch.nn as nn
import torch
import config as cfg


class GPT2(nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()
        # 用于创建字典参数，来对字典里的字进行编码
        self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim)

        # 用于位置编码
        self.positon_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)

        # 用于句子编码
        self.type_embed = nn.Embedding(cfg.type_num, cfg.embed_dim)

        self.block = []
        for _ in range(6):
            self.block.append(Block(True))

        self.drop_layer = nn.Dropout(0.1)
        self.block_layer = nn.Sequential(*self.block)
        # 定义一个输出层，对6层block做输出
        self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_num, bias=False)

    def forward(self, vocab, positon):
        # 词嵌入
        vocab = self.vocab_embed(vocab)
        # 位置嵌入
        positon = self.positon_embed(positon)
        # 嵌入完后加入dropout
        drop = self.drop_layer(vocab + positon)
        # 用多个网络块
        block = self.block_layer(drop)
        # 输出层
        output = self.output_layer(block)
        return output


class Block(nn.Module):
    def __init__(self, isMask=False):
        super(Block, self).__init__()
        # 在通道方面做归一化
        self.layer_normal1 = nn.LayerNorm(cfg.embed_dim)
        self.attention_layer = Attention(isMask)
        self.layer_normal2 = nn.LayerNorm(cfg.embed_dim)

        # 扩大参数量 前馈网络
        self.output_layer = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.head_num * cfg.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.head_num * cfg.embed_dim, cfg.embed_dim)
        )
        self.drop_layer = nn.Dropout(0.1)

    def forward(self, data):
        normal1 = self.layer_normal1(data)
        attention = self.attention_layer(normal1)
        attention = attention + data
        normal2 = self.layer_normal2(attention)
        output = self.output_layer(normal2)
        output = self.drop_layer(output)
        output = output + normal2
        return output


class Attention(nn.Module):
    def __init__(self, isMask=False):
        super(Attention, self).__init__()

        self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5
        self.isMask = isMask
        # 把一个词复制三份，Q,K,V
        self.copy_layer = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)
        self.drop_layer = nn.Dropout(0.1)

        # 在计算完注意力后来个线性层来计算注意力
        self.output_layer = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        if self.isMask:
            # 使用tril删除右上角的数据，并使用register_buffer保存mask状态，使它不在bp算法被当作参数更新，但可以被传入CUAD或CPU进行计算，保存时会被当作权重保存，但不参与训练
            self.register_buffer("mask", torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)))

    def forward(self, data):
        # n,s,768 --> n,s,768*3   # q, k, v
        data = self.copy_layer(data)

        # n,s,768 --> n,s,12,64*3   #　多头 12个头, 相当于就是将data的最后一个维度切分成->(12, -1)
        data = data.reshape(*data.shape[:-1], cfg.head_num, -1)

        # n,s,12,64*3 --> n,12,s,64*3
        data = data.transpose(-2, -3)

        # n,12,s,64*3 --> n,12,s,64
        q, k, v = data.chunk(3, dim=-1)   # 按照最后一个维度进行切分成三分

        # n,12,s,64 --> n,12,s,s
        w = (q @ (k.transpose(-1, -2))) / self.dk

        if self.isMask:
            mask = self.mask[0:w.size(-2), 0:w.size(-1)]
            # 保证右上角无限小，这样经过softmax以后的值为0
            w = w * mask - (1 - mask) * 1e5
        w = torch.softmax(w, dim=-1)
        w = self.drop_layer(w)

        # n,12,s,s-->n,12,s,v(64)
        value = w @ v
        # n,12,s,v(64)-->n,s,12,v(64)
        value = value.transpose(-2, -3)
        # n,12,s,v(64)-->n,s,768
        value = value.reshape(*value.shape[0:-2], -1)
        output = self.output_layer(value)
        output = self.drop_layer(output)
        return output


if __name__ == '__main__':
    vocab = torch.tensor([[0, 1]])
    positon = torch.tensor([[0, 1]])
    # type = torch.tensor([[0]])
    gpt = GPT2()
    gpt.eval()
    y = gpt(vocab, positon)
    print(y.shape)   # torch.Size([1, 2, 2123])

