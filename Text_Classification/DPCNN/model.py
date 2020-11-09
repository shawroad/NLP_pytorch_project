# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/11/07 16:59:48
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        # 1. 词嵌入
        self.embedding = nn.Embedding(args.n_vocab, args.embed_dim, padding_idx=args.n_vocab-2)

        # 2. 卷积
        self.conv_region = nn.Conv2d(1, args.num_filters, (3, args.embed_dim), stride=1)
        self.conv = nn.Conv2d(args.num_filters, args.num_filters, (3, 1), stride=1)

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)

        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))   # top bottom
        self.relu = nn.ReLU()
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))   # bottom
        self.fc = nn.Linear(args.num_filters, args.num_classes)

    def forward(self, input_ids, seq_len, label):
        # input_ids: batch_size, max_len
        x = self.embedding(input_ids)    # torch.Size([32, 512, 300])
        x = x.unsqueeze(1)   # batch_size, 1, max_len, embed_dim

        x = self.conv_region(x)   # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)   # batch_size, 250, seq_len, 1
        x = self.relu(x)

        x = self.conv(x)    # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)    # [batch_size, 250, seq_len, 1]
        x = self.relu(x)

        x = self.conv(x)   # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x
        
    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)
        # Short Cut
        x = x + px
        return x

