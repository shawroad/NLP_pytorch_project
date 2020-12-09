"""
# -*- coding: utf-8 -*-
# @File    : model.py
# @Time    : 2020/12/8 5:24 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
from torch import nn
from pdb import set_trace


def seq_max_pool(x):
    '''
    :param x: x[0]->seq:(batch_size, seq_len, hidden_size)   x[1]->mask:(batch_size, seq_len, 1)
    :return: 先出去mask部分,然后做 maxpooling
    '''
    seq, mask = x
    seq = seq - (1 - mask) * 1e10
    return torch.max(seq, 1)


def seq_and_vec(x):
    '''
    :param x: x[0]->seq:(batch_size, seq_len, hidden_size), x[1]->全局池化的向量: (batch_size, hidden_size)
    :return: 相当于将全局的向量拼接到seq后面
    '''
    seq, vec = x
    vec = torch.unsqueeze(vec, 1)    # (batch_size, 1, hidden_size)
    # print(torch.zeros_like(seq[:, :, :1]).size())   # batch_size, max_len, 1
    vec = torch.zeros_like(seq[:, :, :1]) + vec   # batch_size, max_len, hidden_size
    return torch.cat([seq, vec], 2)


def seq_gather(x):
    '''
    [t, k1]
    # k1是随机采样的一个主体, 从t编码向量中  将k1对应的向量抽取出来  然后添加到对应的位置
    :param x: x[0]->seq: batch_size, max_len, hidden_size; k1: batch_size, 1
    :return:
    '''
    seq, idxs = x
    batch_idxs = torch.arange(0, seq.size(0))
    batch_idxs = torch.unsqueeze(batch_idxs, 1)
    # print(batch_idxs.size())    # (batch_size, 1)
    idxs = torch.cat([batch_idxs, idxs], 1)
    res = []
    for i in range(idxs.size(0)):
        vec = seq[idxs[i][0], idxs[i][1], :]
        res.append(torch.unsqueeze(vec, 0))
    res = torch.cat(res)
    return res


class s_model(nn.Module):
    def __init__(self, word_dict_length, word_emb_size, lstm_hidden_size):
        super(s_model, self).__init__()
        # 1. 词嵌入
        self.embeds = nn.Embedding(word_dict_length, word_emb_size)

        self.dropout = nn.Dropout(0.25)

        self.bilstm = nn.LSTM(input_size=word_emb_size, hidden_size=int(word_emb_size / 2), num_layers=2,
                              batch_first=True, bidirectional=True)

        self.conv = nn.Sequential(nn.Conv1d(in_channels=word_emb_size*2, out_channels=word_emb_size,
                                            kernel_size=3, stride=1, padding=1),
                                  nn.ReLU())
        self.fc1 = nn.Linear(word_emb_size, 1)
        self.fc2 = nn.Linear(word_emb_size, 1)

    def forward(self, t):
        # t.size()   batch_size, max_len, 1
        mask = torch.gt(torch.unsqueeze(t, 2), 0).type(torch.FloatTensor)
        mask.requires_grad = False
        outs = self.embeds(t)
        # print(outs.size())    # torch.Size([2, 126, 128])
        t = self.dropout(outs)
        t = torch.mul(t, mask)   # torch.Size([2, 126, 128])

        t, (h_n, c_n) = self.bilstm(t)
        # print(t.size())    # torch.Size([2, 126, 128])
        t_max, t_max_index = seq_max_pool([t, mask])
        # t_max: (batch_size, hidden_size), t_max_index: (batch_size, hidden_size)

        h = seq_and_vec([t, t_max])   # 相当于把全局池化后的向量拼接到序列中每个token向量之后
        # print(h.size())    # torch.Size([2, 126, 256])
        h = h.permute(0, 2, 1)   # torch.Size([2, 256, 126])
        h = self.conv(h)
        # print(h.size())    # torch.Size([2, 128, 126])
        h = h.permute(0, 2, 1)

        ps1 = self.fc1(h)
        ps2 = self.fc2(h)
        return [ps1, ps2, t, t_max, mask]


class po_model(nn.Module):
    def __init__(self, word_dict_length, word_emb_size, lstm_hidden_size, num_classes):
        super(po_model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=word_emb_size * 4,  # 输入的深度
                out_channels=word_emb_size,  # filter 的个数，输出的高度
                kernel_size=3,  # filter的长与宽
                stride=1,  # 每隔多少步跳一下
                padding=1,  # 周围围上一圈 if stride= 1, pading=(kernel_size-1)/2
            ),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(word_emb_size, num_classes + 1)
        self.fc2 = nn.Linear(word_emb_size, num_classes + 1)

    def forward(self, t, t_max, k1, k2):
        # t:(batch_size, max_len, hidden_size); t_max:(batch_size, hidden_size) k1:(batch_size, 1, k2: batch_size, 1)
        k1 = seq_gather([t, k1])   # 抽取实体的起始向量
        k2 = seq_gather([t, k2])   # 抽取实体的结束向量
        # print(k1.size())    # torch.Size([2, 128])
        # print(k2.size())    # torch.Size([2, 128])
        k = torch.cat([k1, k2], 1)   # torch.Size([2, 256])
        h = seq_and_vec([t, t_max])   # 将上一个模型池化后的向量加入到序列编码的每个token之后
        h = seq_and_vec([h, k])    # 将实体的起始终止杂糅向量 拼接到序列编码的每个token之后
        # print(h.size())   # torch.Size([2, 126, 512])
        h = h.permute(0, 2, 1)
        # print(h.size())   # torch.Size([2, 512, 126])
        h = self.conv(h)
        # print(h.size())    # torch.Size([2, 128, 126])
        h = h.permute(0, 2, 1)    # torch.Size([2, 126, 128])
        po1 = self.fc1(h)
        # print(po1.size())     # torch.Size([2, 126, 50])
        po2 = self.fc2(h)      # torch.Size([2, 126, 50])
        return [po1, po2]
