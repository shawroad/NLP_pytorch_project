"""
# -*- coding: utf-8 -*-
# @File    : BiLSTM_ATT.py
# @Time    : 2020/12/8 11:47 上午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace
from config import set_args


class BiLSTM_ATT(nn.Module):
    def __init__(self, embedding_pre=None):

        super(BiLSTM_ATT, self).__init__()
        args = set_args()
        self.hidden_size = args.hidden_size
        self.tag_size = args.tag_size

        # 1. 词嵌入
        if args.is_train_embedding:
            self.word_embeds = nn.Embedding(args.vocab_size, args.embed_dim)
        else:
            self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre), freeze=False)

        # 两种位置嵌入
        self.pos1_embeds = nn.Embedding(args.pos_size, args.pos_dim)
        self.pos2_embeds = nn.Embedding(args.pos_size, args.pos_dim)

        self.lstm = nn.LSTM(input_size=args.embed_dim + args.pos_dim * 2, hidden_size=args.hidden_size // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)
        self.relation_embeds = nn.Embedding(args.tag_size, self.hidden_size)


    def attention(self, H):
        M = torch.tanh(H)
        batch_size = H.size(0)
        self.att_weight = nn.Parameter(torch.randn(batch_size, 1, self.hidden_size))
        a = F.softmax(torch.bmm(self.att_weight, M), 2)   # torch.Size([8, 1, 200]) x torch.Size([8, 200, 80])
        a = torch.transpose(a, 1, 2)   # torch.Size(8, 80, 1)
        return torch.bmm(H, a)   # torch.Size(8, 200, 1)

    def forward(self, input_ids, pos1, pos2):
        batch_size = input_ids.size(0)
        # 词嵌入
        embeds = torch.cat((self.word_embeds(input_ids), self.pos1_embeds(pos1), self.pos2_embeds(pos2)), dim=-1)

        lstm_out, hidden = self.lstm(embeds)   # 原始的lstm_out
        # last_out size: torch.Size([8, 80, 200])
        # hidden (tuple): len=2  (torch.Size([80, 200]), torch.Size([80, 200]))

        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = self.dropout_lstm(lstm_out)
        # print(lstm_out.size())    # torch.Size([8, 200, 80])
        att_out = torch.tanh(self.attention(lstm_out))
        att_out = self.dropout_att(att_out)    # torch.Size([8, 200, 1])

        relation = torch.tensor([i for i in range(self.tag_size)], dtype=torch.long).repeat(batch_size, 1)
        # print(relation.size())   # torch.Size([8, 12])
        relation = self.relation_embeds(relation)
        # print(relation.size())    #  torch.Size([8, 12, 200])

        self.relation_bias = nn.Parameter(torch.randn(batch_size, self.tag_size, 1))
        res = torch.add(torch.bmm(relation, att_out), self.relation_bias)   # batch_size, 12, 1
        res = F.softmax(res, 1).squeeze(-1)
        return res

