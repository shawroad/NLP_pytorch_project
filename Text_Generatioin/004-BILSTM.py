"""

@file  : 004-BILSTM.py

@author: xiaolu

@time  : 2019-11-08

"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

dtype = torch.FloatTensor


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=n_hidden, bidirectional=True)  # 双向的lstm
        self.W = nn.Parameter(torch.randn([n_hidden * 2, vocab_size], requires_grad=True).type(dtype))
        self.b = nn.Parameter(torch.randn([vocab_size], requires_grad=True).type(dtype))

    def forward(self, X):
        input = X.transpose(0, 1)  # input : [n_step, batch_size, vocab_size]
        # 双向的初始化状态
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        hidden_state = Variable(torch.zeros(1*2, len(X), n_hidden))
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1*2, len(X), n_hidden))

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        # print(outputs.size())   # torch.Size([25, 24, 10])  # 自动将双向的向量进行拼接

        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = torch.mm(outputs, self.W) + self.b  # model : [batch_size, vocab_size]
        return model


def make_batch(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [vocab2id[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input))   # 填充长度
        target = vocab2id[words[i + 1]]
        input_batch.append(np.eye(vocab_size)[input])
        target_batch.append(target)
    return Variable(torch.Tensor(input_batch)), Variable(torch.LongTensor(target_batch))


if __name__ == '__main__':
    sentence = (
        'Lorem ipsum dolor sit amet consectetur adipisicing elit'
        'sed do eiusmod tempor incididunt ut labore et dolore magna'
        'aliqua Ut enim ad minim veniam quis nostrud exercitation'
    )

    vocab2id = {w: i for i, w in enumerate(list(set(sentence.split())))}
    id2vocab = {i: w for i, w in enumerate(list(set(sentence.split())))}
    # print(vocab2id)
    vocab_size = len(vocab2id)
    max_len = len(sentence.split())

    # 把隐层的维度设置成5
    n_hidden = 5


    input_batch, target_batch = make_batch(sentence)

    model = BiLSTM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(10000):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(sentence)
    print([id2vocab[n.item()] for n in predict.squeeze()])
