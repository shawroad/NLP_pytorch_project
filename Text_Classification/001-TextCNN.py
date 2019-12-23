"""

@file  : 001-TextCNN.py

@author: xiaolu

@time  : 2019-11-08

"""
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

dtype = torch.FloatTensor


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        self.num_filters_total = num_filters * len(filter_sizes)

        # 类似与embedding
        # self.W = nn.Parameter(torch.empty(vocab_size, embedding_size).uniform_(-1, 1), requires_grad=True).type(dtype)
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # 定义全连接的权重
        self.Weight = nn.Parameter(torch.empty(self.num_filters_total, num_classes).uniform_(-1, 1), requires_grad=True).type(dtype)
        self.Bias = nn.Parameter(0.1 * torch.ones([num_classes]), requires_grad=True).type(dtype)

    # def forward(self, X):
    def forward(self, X):

        # embedded_chars = self.W[X]  # [batch_size, sequence_length, embedding_size]
        embedded_chars = self.embedding(X)

        # 在第一维度加个通道　为了卷积[batch, channel(=1), sequence_length, embedding_size]
        embedded_chars = embedded_chars.unsqueeze(1)

        pooled_outputs = []
        for filter_size in filter_sizes:

            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            conv = nn.Conv2d(1, num_filters, (filter_size, embedding_size), bias=True)(embedded_chars)
            h = F.relu(conv)

            # mp : ((filter_height, filter_width))
            mp = nn.MaxPool2d((sequence_length - filter_size + 1, 1))

            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)

        # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool = torch.cat(pooled_outputs, len(filter_sizes))

        # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])

        # 类似全连接
        model = torch.mm(h_pool_flat, self.Weight) + self.Bias  # [batch_size, num_classes]
        return model


if __name__ == "__main__":
    # 制造语料和标签
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

    # 超参数
    embedding_size = 2  # 词嵌入
    sequence_length = 3   # 序列长度
    num_classes = 2  # 0 or 1
    filter_sizes = [2, 2, 2]   # 卷积核的大小
    num_filters = 3   # 卷积核的个数　　

    # 数据的预处理
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    vocab2id = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(vocab2id)

    inputs = []
    for sen in sentences:
        inputs.append(np.asarray([vocab2id[n] for n in sen.split()]))

    targets = []
    for out in labels:
        targets.append(out)   # To using Torch Softmax Loss function

    input_batch = Variable(torch.LongTensor(inputs))
    target_batch = Variable(torch.LongTensor(targets))

    model = TextCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Test
    test_text = 'sorry hate you'
    tests = [np.asarray([vocab2id[n] for n in test_text.split()])]
    test_batch = Variable(torch.LongTensor(tests))

    # Predict
    predict = model(test_batch).data.max(1, keepdim=True)[1]

    if predict[0][0] == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")
