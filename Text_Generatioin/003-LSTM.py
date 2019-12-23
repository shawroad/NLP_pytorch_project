"""

@file  : 003-LSTM.py

@author: xiaolu

@time  : 2019-11-08

"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype = torch.FloatTensor


class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=char_size, hidden_size=n_hidden)

        # 外加一个全连接
        self.W = nn.Parameter(torch.randn([n_hidden, char_size]).type(dtype))
        self.b = nn.Parameter(torch.randn([char_size]).type(dtype))

    def forward(self, X):
        input = X.transpose(0, 1)  # X: [batch_size, n_step, n_class] -> [n_step, batch_size, char_size]

        # 给隐层一个初始化 由于是lstm  所以包含cell 和　hidden
        # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        hidden_state = Variable(torch.zeros(1, len(X), n_hidden))
        # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1, len(X), n_hidden))

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))

        outputs = outputs[-1]  # [batch_size, n_hidden]

        model = torch.mm(outputs, self.W) + self.b  # model : [batch_size, n_class]

        return model


def make_batch(seq_data):
    input_batch, target_batch = [], []

    for seq in seq_data:
        input = [char2id[n] for n in seq[:-1]]   # 'm', 'a' , 'k' is input
        target = char2id[seq[-1]]  # 'e' is target
        input_batch.append(np.eye(char_size)[input])
        target_batch.append(target)
    return Variable(torch.Tensor(input_batch)), Variable(torch.LongTensor(target_batch))


if __name__ == '__main__':
    # 构建字符表
    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    char2id = {n: i for i, n in enumerate(char_arr)}
    id2char = {i: w for i, w in enumerate(char_arr)}
    char_size = len(char2id)

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

    # 超参数
    # n_step = 3
    n_hidden = 128  # 隐层的维度

    input_batch, target_batch = make_batch(seq_data)
    model = TextLSTM()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    output = model(input_batch)

    # Training
    for epoch in range(1000):
        optimizer.zero_grad()

        output = model(input_batch)
        loss = criterion(output, target_batch)

        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    inputs = [sen[:3] for sen in seq_data]

    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(inputs, '->', [id2char[n.item()] for n in predict.squeeze()])
