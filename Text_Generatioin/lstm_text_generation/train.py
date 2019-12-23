"""

@file  : train.py

@author: xiaolu

@time  : 2019-11-07

"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus


# 建立基础模型
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # 词嵌入
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # 将输入编码为向量
        x = self.embed(x)

        # 输入到lstm中
        out, (h, c) = self.lstm(x, h)
        # print(out.size())   # torch.Size([20, 30, 1024])

        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        print(out.size())    # torch.Size([600, 1024])

        out = self.linear(out)
        return out, (h, c)


def detach(states):
    # Truncated backpropagation 截断方向误差传播
    return [state.detach() for state in states]


if __name__ == "__main__":
    # 查看当前环境是否有可用的cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 超参数的定义
    embed_size = 128
    hidden_size = 1024
    num_layers = 1
    num_epochs = 5
    num_samples = 1000  # number of words to be sampled
    batch_size = 20
    seq_length = 30
    learning_rate = 0.002

    # 加载数据集　并建立词典
    corpus = Corpus()
    ids = corpus.get_data('data/train.txt', batch_size)
    # print(ids.size())  # torch.Size([20, 46479])
    vocab_size = len(corpus.dictionary)
    num_batches = ids.size(1) // seq_length

    model = RNN(vocab_size, embed_size, hidden_size, num_layers).to(device)   # 如果有用的gpu直接放gpu上运行

    # 定义loss 以及优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))
        for i in range(0, ids.size(1) - seq_length, seq_length):
            # 知道前sequence个单词　预测往后移动一位的sequence个单词
            inputs = ids[:, i: i+seq_length].to(device)
            targets = ids[:, (i+1): (i+1) + seq_length].to(device)

            states = detach(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))  # 输出是one_hot 但是真实标签我们不用转one_hot

            # 反向传播＋优化
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪

            optimizer.step()

            step = (i + 1) // seq_length

            if step % 1 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

    # 测试模型
    with torch.no_grad():
        with open('sample.txt', 'w') as f:
            state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                     torch.zeros(num_layers, 1, hidden_size).to(device))

            # 选择一个随机的id
            prob = torch.ones(vocab_size)
            input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

            for i in range(num_samples):
                output, state = model(input, state)

                # Sample a word id
                prob = output.exp()
                word_id = torch.multinomial(prob, num_samples=1).item()

                # Fill input with sampled word id for the next time step
                input.fill_(word_id)

                word = corpus.dictionary.idx2word[word_id]
                word = '\n' if word == '<eos>' else word + ' '
                f.write(word)

                if (i + 1) % 100 == 0:
                    print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))

    # 保存模型
    torch.save(model.state_dict(), 'model.ckpt')
