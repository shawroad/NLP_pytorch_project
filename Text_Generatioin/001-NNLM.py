"""

@file  : 001-NNLM.py

@author: xiaolu

@time  : 2019-11-08

"""
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

# 类型
dtype = torch.FloatTensor

# 是否有可用的cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NNLM(nn.Module):
    '''
    定义模型
    '''
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(vocab_size, m)

        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype), requires_grad=True)
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype), requires_grad=True)

        self.W = nn.Parameter(torch.randn(n_step * m, vocab_size).type(dtype), requires_grad=True)
        self.b = nn.Parameter(torch.randn(vocab_size).type(dtype), requires_grad=True)

        self.U = nn.Parameter(torch.randn(n_hidden, vocab_size).type(dtype), requires_grad=True)

    def forward(self, X):
        '''
        思想: 1: 想文章进行词嵌入, 然后将文章中的所有词向量拼接
             2: 分两路走 一路是直接一个全连接映射到词表
             3: 第二个分支 是将第一步的词向量拼接完成后 先经过一个隐藏层 接着激活 最后在经过一个全连接 映射到词表
             4: 将二三的结果对应位置进行拼接得到最后的输出
        '''
        # 1. 词嵌入
        X = self.C(X)

        # 2. 将一句话中的所有词的词向量拼接到一块
        X = X.view(-1, n_step * m)   # # [batch_size, n_step * m]
        tanh = torch.tanh(self.d + torch.mm(X, self.H))   # [batch_size, n_hidden]

        output = self.b + torch.mm(X, self.W) + torch.mm(tanh, self.U)  # [batch_size, vocab_size]

        return output


def make_batch(sentences):
    # 制造批数据
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word2id[n] for n in word[:-1]]
        target = word2id[word[-1]]

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch


if __name__ == '__main__':
    # 自己制造语料
    sentences = ["i like dog", "i love coffee", "i hate milk"]

    # 简单进行语料处理
    word_list = " ".join(sentences).split()  # 词未去重的列表
    word_list = list(set(word_list))
    word2id = {w: i for i, w in enumerate(word_list)}
    id2word = {i: w for i, w in enumerate(word_list)}

    vocab_size = len(word2id)

    # 一些超参数
    n_step = 2
    n_hidden = 2  # 隐层的维度
    m = 2   # 词嵌入的维度

    model = NNLM()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch(sentences)
    input_batch = Variable(torch.LongTensor(input_batch))
    target_batch = Variable(torch.LongTensor(target_batch))

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]

    # Test
    print([sen.split()[:2] for sen in sentences], '->', [id2word[n.item()] for n in predict.squeeze()])
