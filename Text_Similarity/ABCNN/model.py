"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-07
"""
import torch
from torch import nn
import torch.nn.functional as F
from pdb import set_trace


class WideConv(nn.Module):
    def __init__(self, seq_len, embeds_size):
        super(WideConv, self).__init__()
        self.seq_len = seq_len
        self.embeds_size = embeds_size
        self.W = nn.Parameter(torch.randn((seq_len, embeds_size)))
        nn.init.xavier_normal_(self.W)
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.tanh = nn.Tanh()

    def forward(self, sent1, sent2, mask1, mask2):
        # print(mask1.size())   # torch.Size([2, 40])
        # print(mask2.size())   # torch.Size([2, 40])
        A = match_score(sent1, sent2, mask1, mask2)   # torch.Size([2, 40, 40])

        attn_feature_map1 = A.matmul(self.W)    # torch.Size([2, 40, 300])

        attn_feature_map2 = A.transpose(1, 2).matmul(self.W)   # torch.Size([2, 40, 300])

        x1 = torch.cat([sent1.unsqueeze(1), attn_feature_map1.unsqueeze(1)], 1)    # torch.Size([2, 2, 40, 300])
        x2 = torch.cat([sent2.unsqueeze(1), attn_feature_map2.unsqueeze(1)], 1)    # torch.Size([2, 2, 40, 300])
        o1, o2 = self.conv(x1).squeeze(1), self.conv(x2).squeeze(1)
        o1, o2 = self.tanh(o1), self.tanh(o2)   # torch.Size([2, 40, 300]), torch.Size([2, 40, 300])
        return o1, o2


class Model(nn.Module):
    def __init__(self, embeddings, num_layer=1, linear_size=300, max_length=40):
        super(Model, self).__init__()
        self.linear_size = linear_size
        self.num_layer = num_layer
        self.embeds_dim = embeddings.shape[1]

        # 1. 词嵌入
        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings).float(), requires_grad=True)

        # 2. 卷积
        self.conv = nn.ModuleList([WideConv(max_length, embeddings.shape[1]) for _ in range(self.num_layer)])

        self.fc = nn.Sequential(
            nn.Linear(self.embeds_dim * (1 + self.num_layer) * 2, self.linear_size),
            nn.LayerNorm(self.linear_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear_size, 2),
        )

    def forward(self, q1, q2):
        mask1, mask2 = q1.eq(0), q2.eq(0)
        res = [[], []]
        q1_encode = self.embed(q1)   # torch.Size([2, 40, 300])
        q2_encode = self.embed(q2)   # torch.Size([2, 40, 300])

        # 分别对q1_encode和q2_encode进行平均池化
        res[0].append(F.avg_pool1d(q1_encode.transpose(1, 2), kernel_size=q1_encode.size(1)).squeeze(-1))  # (2, 300)
        res[1].append(F.avg_pool1d(q2_encode.transpose(1, 2), kernel_size=q2_encode.size(1)).squeeze(-1))  # (2, 300)

        for i, conv in enumerate(self.conv):
            o1, o2 = conv(q1_encode, q2_encode, mask1, mask2)
            res[0].append(F.avg_pool1d(o1.transpose(1, 2), kernel_size=o1.size(1)).squeeze(-1))
            res[1].append(F.avg_pool1d(o2.transpose(1, 2), kernel_size=o2.size(1)).squeeze(-1))

            o1, o2 = attention_avg_pooling(o1, o2, mask1, mask2)
            q1_encode, q2_encode = o1 + q1_encode, o2 + q2_encode

        x = torch.cat([torch.cat(res[0], 1), torch.cat(res[1], 1)], 1)
        # print(x.size())   # torch.Size([2, 1200])

        sim = self.fc(x)   # torch.Size([2, 2])
        probabilities = torch.softmax(sim, dim=-1)   # torch.Size([2, 2])

        return sim, probabilities


def match_score(s1, s2, mask1, mask2):
    '''
    :param s1: p   size: batch_size, seq_len, dim
    :param s2: h   size: batch_size, seq_len, dim
    :param mask1:   size: batch_size, seq_len
    :param mask2:   size: batch_size, seq_len
    :return:
    '''
    batch_size, seq_len, dim = s1.shape

    s1 = s1 * mask1.eq(0).unsqueeze(2).float()   # torch.Size([2, 40, 300])
    s2 = s2 * mask2.eq(0).unsqueeze(2).float()   # torch.Size([2, 40, 300])

    s1 = s1.unsqueeze(2).repeat(1, 1, seq_len, 1)   # torch.Size([2, 40, 40, 300])
    s2 = s2.unsqueeze(1).repeat(1, seq_len, 1, 1)   # torch.Size([2, 40, 40, 300])
    a = s1 - s2
    a = torch.norm(a, dim=-1, p=2)
    return 1.0 / (1.0 + a)


def attention_avg_pooling(sent1, sent2, mask1, mask2):
    '''
    :param sent1:  size: batch_size, seq_len, dim
    :param sent2:
    :param mask1:  size: batch_size, seq_len
    :param mask2:
    :return:
    '''
    A = match_score(sent1, sent2, mask1, mask2)
    weight1 = torch.sum(A, dim=-1)
    weight2 = torch.sum(A.transpose(1, 2), dim=-1)
    s1 = sent1 * weight1.unsqueeze(2)
    s2 = sent2 * weight2.unsqueeze(2)
    s1 = F.avg_pool1d(s1.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s2 = F.avg_pool1d(s2.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s1, s2 = s1.transpose(1, 2), s2.transpose(1, 2)
    return s1, s2