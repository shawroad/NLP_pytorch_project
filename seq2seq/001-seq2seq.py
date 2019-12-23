"""

@file  : 001-seq2seq.py

@author: xiaolu

@time  : 2019-11-08

"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

dtype = torch.FloatTensor


class Seq2Seq(nn.Module):
    '''
    建立模型
    '''
    def __init__(self):
        super(Seq2Seq, self).__init__()

        self.enc_cell = nn.RNN(input_size=char_size, hidden_size=n_hidden, dropout=0.5)
        self.dec_cell = nn.RNN(input_size=char_size, hidden_size=n_hidden, dropout=0.5)

        self.fc = nn.Linear(n_hidden, char_size)

    def forward(self, enc_input, enc_hidden, dec_input):

        enc_input = enc_input.transpose(0, 1)  # enc_input: [max_len(=n_step, time step), batch_size, n_class]
        dec_input = dec_input.transpose(0, 1)  # dec_input: [max_len(=n_step, time step), batch_size, n_class]

        # enc_states : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, enc_states = self.enc_cell(enc_input, enc_hidden)

        # outputs : [max_len+1(=6), batch_size, num_directions(=1) * n_hidden(=128)]
        outputs, _ = self.dec_cell(dec_input, enc_states)   # 也就是编码的最后一步的隐层状态输入到解码的隐态初始化的隐层

        model = self.fc(outputs)  # model : [max_len+1(=6), batch_size, n_class]

        return model


def translate(word):

    input_batch, output_batch, _ = make_batch([[word, 'P' * len(word)]])
    # make hidden shape [num_layers * num_directions, batch_size, n_hidden]

    hidden = Variable(torch.zeros(1, 1, n_hidden))
    output = model(input_batch, hidden, output_batch)
    # output : [max_len+1(=6), batch_size(=1), n_class]

    predict = output.data.max(2, keepdim=True)[1]  # select n_class dimension
    decoded = [char_arr[i] for i in predict]
    end = decoded.index('E')   # 找出结束标志　然后截取结束标志之前的文本
    translated = ''.join(decoded[:end])

    return translated.replace('P', '')


def make_batch(seq_data):
    input_batch, output_batch, target_batch = [], [], []
    for seq in seq_data:
        for i in range(2):  # 填充输入和输出
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))

        input = [char2id[n] for n in seq[0]]
        output = [char2id[n] for n in ('S' + seq[1])]  # 解码的输入 加开始标志
        target = [char2id[n] for n in (seq[1] + 'E')]  # 加输出结束的标志

        input_batch.append(np.eye(char_size)[input])  # 转化为对应的one_hot
        output_batch.append(np.eye(char_size)[output])   # 转化为对应的one_hot
        target_batch.append(target)   # not one-hot

    # make tensor
    return Variable(torch.Tensor(input_batch)), Variable(torch.Tensor(output_batch)), Variable(torch.LongTensor(target_batch))


if __name__ == '__main__':
    # S: 开始
    # E: 结束
    # P: 填充
    # 1. 构造数据　并整理词表　
    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    char2id = {n: i for i, n in enumerate(char_arr)}

    # 参数
    n_step = 5   # 即我们常说的maxlen
    n_hidden = 128
    char_size = len(char2id)
    batch_size = len(seq_data)

    input_batch, output_batch, target_batch = make_batch(seq_data)

    model = Seq2Seq()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5000):
        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = Variable(torch.zeros(1, batch_size, n_hidden))   # 隐层初始化

        optimizer.zero_grad()

        # input_batch : [batch_size, max_len(=n_step, time step), n_class]
        # output_batch : [batch_size, max_len+1(=n_step, time step) (becase of 'S' or 'E'), n_class]
        # target_batch : [batch_size, max_len+1(=n_step, time step)], not one-hot
        output = model(input_batch, hidden, output_batch)

        # output : [max_len+1, batch_size, num_directions(=1) * n_hidden]
        output = output.transpose(0, 1)  # [batch_size, max_len+1(=6), num_directions(=1) * n_hidden]

        loss = 0
        for i in range(0, len(target_batch)):
            # output[i] : [max_len+1, num_directions(=1) * n_hidden, target_batch[i] : max_len+1]
            loss += criterion(output[i], target_batch[i])

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        # if (epoch + 1) % 1000 == 0:
        #     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))