"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-08
"""
import torch
import torch.nn as nn
from modules import Module, ModuleList, ModuleDict
from modules.encoder import Encoder
from modules.alignment import registry as alignment
from modules.fusion import registry as fusion
from modules.connection import registry as connection
from modules.pooling import Pooling
from modules.prediction import registry as prediction


class Model(Module):
    def __init__(self, args, embeddings):
        super(Model, self).__init__()
        self.dropout = args.dropout
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings).float(), requires_grad=True)

        self.blocks = ModuleList([ModuleDict({
            'encoder': Encoder(args, args.embedding_dim if i == 0 else args.embedding_dim + args.hidden_size),
            'alignment': alignment[args.alignment](
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
            'fusion': fusion[args.fusion](
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
        }) for i in range(args.blocks)])

        self.connection = connection[args.connection]()
        self.pooling = Pooling()
        self.prediction = prediction[args.prediction](args)

    def forward(self, a, b):
        mask_a = torch.ne(a, 0).unsqueeze(2)
        mask_b = torch.ne(b, 0).unsqueeze(2)
        a = self.embedding(a)
        b = self.embedding(b)
        res_a, res_b = a, b

        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a)
            b_enc = block['encoder'](b, mask_b)
            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        logits = self.prediction(a, b)
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities
