"""

@file  : 002-RNN.py

@author: xiaolu

@time  : 2019-11-08

"""
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

dtype = torch.FloatTensor


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()

        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.randn([n_hidden, vocab_size], requires_grad=True).type(dtype))
        self.b = nn.Parameter(torch.randn([vocab_size], requires_grad=True).type(dtype))

    def forward(self, hidden, X):
        # X : [n_step, batch_size, n_class]
        X = X.transpose(0, 1)

        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        outputs = outputs[-1]  # [batch_size, num_directions(=1) * n_hidden]
        model = torch.mm(outputs, self.W) + self.b  # model : [batch_size, n_class]
        return model


def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [vocab2id[n] for n in word[:-1]]
        target = vocab2id[word[-1]]

        input_batch.append(np.eye(vocab_size)[input])
        target_batch.append(target)
    return input_batch, target_batch


if __name__ == '__main__':
    # 构造语料
    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()

    word_list = list(set(word_list))
    vocab2id = {w: i for i, w in enumerate(word_list)}
    id2vocab = {i: w for i, w in enumerate(word_list)}
    vocab_size = len(vocab2id)

    # 超参数
    batch_size = len(sentences)
    # n_step = 2  # number of cells(= number of Step)
    n_hidden = 5  # 隐层维度

    input_batch, target_batch = make_batch(sentences)
    input_batch = Variable(torch.Tensor(input_batch))
    target_batch = Variable(torch.LongTensor(target_batch))

    model = TextRNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5000):
        optimizer.zero_grad()

        # 给一个初始化隐层　hidden : [num_layers * num_directions, batch, hidden_size]
        hidden = Variable(torch.zeros(1, batch_size, n_hidden))

        # input_batch : [batch_size, n_step, n_class]
        output = model(hidden, input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    input = [sen.split()[:2] for sen in sentences]

    # Predict
    hidden = Variable(torch.zeros(1, batch_size, n_hidden))
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [id2vocab[n.item()] for n in predict.squeeze()])
