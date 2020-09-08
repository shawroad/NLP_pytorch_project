"""

@file  : 004-NPLM.py

@author: xiaolu

@time  : 2019-12-23

"""
import jieba
import re
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
from torch import optim
import json
from torch.autograd import Variable


class NPLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NPLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embedding = self.embed(inputs)
        batch_size = embedding.size(0)
        # 接着将两个词的向量进行拼接
        embedding = embedding.view(batch_size, -1)

        out = self.linear1(embedding)
        out = F.relu(out)

        out = self.linear2(out)

        log_probs = F.log_softmax(out, dim=-1)
        return log_probs

    def extract(self, inputs):
        '''
        获取词向量
        :param inputs:
        :return:
        '''
        embedding = self.embed(inputs)
        return embedding


class DataTxt(Dataset):
    def __init__(self, data, vocab2id):
        self.data = data
        self.vocab2id = vocab2id

    def __getitem__(self, item):
        self.x = self.data[item][0]
        self.y = self.data[item][1]

        # 将词组转为id
        self.x = [self.vocab2id.get(self.x[0]), self.vocab2id.get(self.x[1])]
        self.y = [self.vocab2id.get(self.y)]
        self.x = torch.LongTensor(self.x)
        self.y = torch.LongTensor(self.y)
        return self.x, self.y

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # 1. 加载数据
    with open('./data/三体.txt', 'r') as f:
        data = f.read()

    # 2. 分词 并过滤掉标点符号
    temp = jieba.lcut(data)
    words = []
    for i in temp:
        i = i.strip()
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\']《 》+|[+——！，。？、~@#￥%……&*（）]+", '', i)
        if len(i) != 0:
            words.append(i)
    # print(words)

    # 构造三元组 形成三元组
    trigrams = [([[words[i], words[i+1]], words[i+2]]) for i in range(len(words)-2)]
    # print(trigrams)

    # 建立词典
    vocab = list(set(words))
    vocab2id = {}
    vocab2id['<UNK>'] = 0
    for i, v in enumerate(vocab):
        vocab2id[v] = i+1
    id2vocab = {}
    for v, i in vocab2id.items():
        id2vocab[i] = v

    # 定义一个数据加载器
    datatxt = DataTxt(trigrams, vocab2id)
    dataloader = DataLoader(datatxt, shuffle=True, batch_size=64)

    losses = []

    criterion = nn.NLLLoss()
    model = NPLM(len(vocab2id), embedding_dim=128, context_size=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch = 20
    k = 0
    for i in range(epoch):
        total_loss = 0.0
        for x, y in dataloader:
            k += 1
            # print(x)
            y = torch.squeeze(y)
            # print(y)

            optimizer.zero_grad()

            log_prob = model(x)
            loss = criterion(log_prob, y)

            loss.backward()
            optimizer.step()

            total_loss += loss
            print("当前epoch:{}, 当前步:{}, 损失:{}".format(i, k, loss))

        print("平均损失:{}".format(total_loss / len(dataloader)))

        losses.append(total_loss / len(dataloader))
    json.dump(open('loss.json', 'r'), losses)

    vec = model.extract(Variable(torch.LongTensor([v[0] for v in vocab2id.values()])))
    vec = vec.data.numpy()
    print(vec)



