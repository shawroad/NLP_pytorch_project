"""

@file  : 001-skipgram-word2vec.py

@author: xiaolu

@time  : 2019-11-08

"""
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

dtype = torch.FloatTensor


class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()

        # 相当于两次全连接映射
        self.W = nn.Parameter(-2 * torch.rand(vocab_size, embedding_size) + 1, requires_grad=True).type(dtype)
        self.WT = nn.Parameter(-2 * torch.rand(embedding_size, vocab_size) + 1, requires_grad=True).type(dtype)

    def forward(self, X):
        # X: [batch_size, vocab_size]
        hidden_layer = torch.matmul(X, self.W)  # hidden_layer : [batch_size, embedding_size]
        output_layer = torch.matmul(hidden_layer, self.WT)  # output_layer : [batch_size, voc_size]
        return output_layer


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)  # 随机抽取一些索引

    for i in random_index:
        random_inputs.append(np.eye(vocab_size)[data[i][0]])  # 中心词 是以one_hot的形式输入
        random_labels.append(data[i][1])   # 上下文单词
    return random_inputs, random_labels


if __name__ == '__main__':
    sentences = [
        "i like dog",
        "i like cat",
        "i like animal",
        "dog cat animal",
        "apple cat dog like",
        "dog fish milk like",
        "dog cat eyes like",
        "i like apple",
        "apple i hate",
        "apple i movie book music like",
        "cat dog hate",
        "cat dog like"
    ]

    # 预处理
    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))

    vocab2id = {w: i for i, w in enumerate(word_list)}

    # 超参数
    batch_size = 20
    embedding_size = 2  # To show 2 dim embedding graph
    vocab_size = len(vocab2id)

    # Make skip gram of one size window
    skip_grams = []

    for i in range(1, len(word_sequence) - 1):
        target = vocab2id[word_sequence[i]]
        # 这里设置的窗口为１
        context = [vocab2id[word_sequence[i-1]], vocab2id[word_sequence[i+1]]]

        for w in context:
            skip_grams.append([target, w])

    model = Word2Vec()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5000):
        input_batch, target_batch = random_batch(skip_grams, batch_size)

        input_batch = Variable(torch.Tensor(input_batch))
        target_batch = Variable(torch.LongTensor(target_batch))

        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, vocab_size], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()

        x, y = float(W[i][0]), float(W[i][1])
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
