# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/11/05 11:40:05
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        num_classes = 2
        self.embedding = nn.Embedding(args.n_vocab, args.embed_dim, padding_idx=args.n_vocab-2)
        self.embedding_ngram2 = nn.Embedding(args.n_gram_vocab, args.embed_dim)
        self.embedding_ngram3 = nn.Embedding(args.n_gram_vocab, args.embed_dim)

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(args.embed_dim * 3, args.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(args.hidden_size, num_classes)

    def forward(self, input_ids, bigram, trigram, seq_len, label=None):
        word_embedding = self.embedding(input_ids)
        bigram_embedding = self.embedding_ngram2(bigram) 
        trigram_embedding = self.embedding_ngram3(trigram)
        out = torch.cat((word_embedding, bigram_embedding, trigram_embedding), -1)
        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

