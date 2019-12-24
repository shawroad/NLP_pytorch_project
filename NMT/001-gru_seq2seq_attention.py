"""

@file  : 001-gru_seq2seq_attention.py

@author: xiaolu

@time  : 2019-12-23

"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import re
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random


class EncoderRNN(nn.Module):
    '''
    编码器
    输入: (batch_size, seq_length)
    输出: output(每步的输出),　每个隐层向量
    '''
    def __init__(self, input_size, hidden_size, n_layers=1):
        '''
        :param input_size: 词表的大小
        :param hidden_size: 隐层向量的维度
        :param n_layers: 几层
        '''
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        # embedding
        self.embedding = nn.Embedding(input_size, hidden_size)

        # 双向gru
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=self.n_layers, bidirectional=True)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        # print(embedded.size())   # torch.Size([batch_size, seq_length, hidden_size]) 这里hidden_size == embedding_dim

        output = embedded

        output, hidden = self.gru(output, hidden)
        # print(output.size())    # torch.Size([batch_size, seq_length, n_layers * hidden_size])
        # print(hidden.size())    # torch.Size([层数＋方向数, batch_size, hidden_size])
        return output, hidden

    def initHidden(self, batch_size):
        '''
        对隐藏单元进行初始化
        :param batch_size:
        :return:
        '''
        result = Variable(torch.zeros(self.n_layers * 2, batch_size, self.hidden_size))
        return result


class AtteDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=10):

        super(AtteDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size  # 输出词表的大小
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 词嵌入
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # 注意力网络　# 这里之所以2 * n_layers + 1 是两个方向的隐向量＋词嵌入向量
        self.attn = nn.Linear(self.hidden_size * (2 * n_layers + 1), self.max_length)

        # 注意力机制作用后的结果映射到后面的层
        self.attn_combine = nn.Linear(self.hidden_size * 3, self.hidden_size)

        # dropout层
        self.dropout = nn.Dropout(self.dropout_p)

        # 定义一个双向的GRU, 并设置batch_first为True以方便那操作
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=True, num_layers=self.n_layers, batch_first=True)

        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        '''
        :param inputs: decoder_input
        :param hidden: decoder_hidden
        :param encoder_outputs: encoder_outputs
        :return:
        '''
        # 对解码的输入进行词嵌入
        embedded = self.embedding(inputs)  # batch_size x seq_length x embedding_dim

        # 取出初步的隐层
        embedded = embedded[:, 0, :]   # batch_size x hidden_size
        embedded = self.dropout(embedded)

        # 将hidden张量数据转化成batch_size排在第0维度　即:(2*n_layers, batch_size, hidden_size)
        temp_for_transpose = torch.transpose(hidden, 0, 1).contiguous()
        temp_for_transpose = temp_for_transpose.view(temp_for_transpose.size()[0], -1)  # 把隐层向量拉直
        hidden_attn = temp_for_transpose

        # 注意力层的输入 两个方向的hidden_attn 再加上 embedded　
        input_to_attention = torch.cat((embedded, hidden_attn), 1)  # batch_size x hidden_size * (1+direction*n_layers)

        attn_weight = F.softmax(self.attn(input_to_attention))  # batch_size x max_length

        # 当输入数据不标准的话, 对weight截取必要的一段
        attn_weight = attn_weight[:, : encoder_outputs.size()[1]]

        attn_weight = attn_weight.unsqueeze(1)  # batch_size x 1 x max_length

        attn_appiled = torch.bmm(attn_weight, encoder_outputs)  # batch_size x 1 x hidden_size * direction

        # 将输入的词向量与注意力向量作用后的结果拼接成一个大的输入向量
        output = torch.cat((embedded, attn_appiled[:, 0, :]), 1)

        output = self.attn_combine(output).unsqueeze(1)  # batch_size x length_seq x hidden_size

        output = F.relu(output)

        output = self.dropout(output)

        # 开始解码gru的运算
        output, hidden = self.gru(output, hidden)

        # output: batch_size, length_seq, hidden_size*directions
        # hidden: n_layers * directions, batch_size, hidden_size

        # 取出gru运算最后一步的结果,输入给最后一层全连接
        output = self.out(output[:, -1, :])  # batch_size x output_size

        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weight

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers * 2, batch_size, self.hidden_size))
        return result


def clean_data(data):
    data = str(data)
    data = data.lower().strip()
    data = re.sub(r"[.!?]]", r" ", data)
    return data


def build_vocab(data):
    vocab = data.split(' ')
    vocab = list(set(vocab))
    vocab2id = {}
    vocab2id['<PAD>'] = 0
    vocab2id['<UNK>'] = 1

    id2vocab = {}
    id2vocab[0] = '<PAD>'
    id2vocab[1] = '<UNK>'
    for i, v in enumerate(vocab):
        vocab2id[v] = i+2
        id2vocab[i+2] = v

    return vocab2id, id2vocab


def build_vocab2(data):
    vocab = data.split(' ')
    vocab = list(set(vocab))

    vocab2id = {}
    vocab2id['<PAD>'] = 0
    vocab2id['<UNK>'] = 1
    vocab2id['<SOS>'] = 2
    vocab2id['<END>'] = 3

    id2vocab = {}
    id2vocab[0] = '<PAD>'
    id2vocab[1] = '<UNK>'
    id2vocab[2] = '<SOS>'
    id2vocab[3] = '<END>'

    for i, v in enumerate(vocab):
        vocab2id[v] = i + 4
        id2vocab[i + 4] = v

    return vocab2id, id2vocab


def padding_sen(data, max_length):
    result_data = []
    for s in data:
        if len(s) > max_length:
            result_data.append(s[:max_length])
        else:
            result_data.append(s + [0] * (max_length - len(s)))
    return result_data


class DataTxt(Dataset):
    def __init__(self, eng_id_data, fra_id_data):
        self.x = eng_id_data
        self.y = fra_id_data

    def __getitem__(self, item):
        self.x_data = self.x[item]
        self.y_data = self.y[item]
        self.x_data = np.array(self.x_data).astype(np.long)
        self.y_data = np.array(self.y_data).astype(np.long)
        return self.x_data, self.y_data

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    # 加载数据
    with open('./data/fin.txt', 'r') as f:
        lines = f.readlines()
        english = []
        franch = []
        for line in lines:
            eng, fra, _ = line.split('	')
            eng = clean_data(eng)
            english.append(eng)

            fra = clean_data(fra)
            franch.append(fra)

    # 构建英文词典
    str_eng = ' '.join(english)
    eng_vocab2id, eng_id2vocab = build_vocab(str_eng)

    # 构建法文词典
    str_fra = ' '.join(franch)
    fra_vocab2id, fra_id2vocab = build_vocab2(str_fra)
    # print(len(fra_vocab2id))

    # 将各种文转为id序列
    eng_id_data = [[eng_vocab2id.get(v, '<UNK>') for v in sen.split(' ')] for sen in english]

    fra_id_data = [[fra_vocab2id.get(v, '<UNK>') for v in sen.split(' ')] for sen in franch]


    max_length = 15
    eng_id_data = pad_sequences(eng_id_data, maxlen=max_length, value=0, padding='post')
    fra_id_data = pad_sequences(fra_id_data, maxlen=max_length, value=0, padding='post')

    # 数据加载器
    txt = DataTxt(eng_id_data, fra_id_data)
    dataloader = DataLoader(txt, batch_size=64, shuffle=True)

    hidden_size = 32
    n_layers = 2
    eng_vocab_size = len(eng_vocab2id)
    fra_vocab_size = len(fra_vocab2id)

    # 实例化编码器和解码器
    encoder = EncoderRNN(eng_vocab_size, hidden_size, n_layers)
    decoder = AtteDecoderRNN(hidden_size, fra_vocab_size, dropout_p=0.5, max_length=max_length, n_layers=n_layers)

    learning_rate = 0.0001
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    teacher_forcing_ratio = 0.5

    num_epoch = 100

    plot_loss = []
    k = 0
    for epoch in range(num_epoch):
        # 将解码器置于训练状态, 让dropout工作
        decoder.train()
        print_loss_total = 0

        for e, f in dataloader:
            k += 1
            input_variable = torch.LongTensor(e)
            target_variable = torch.LongTensor(f)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_hidden = encoder.initHidden(len(e))

            loss = 0
            # 编码器开始工作
            encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
            # print(encoder_outputs.size())  # torch.Size([64, 15, 64]) batch_size x max_len x hidden_size * bidirection
            # print(encoder_hidden.size())  # torch.Size([4, 64, 32]) n_layers * bidirection x batch_size  x hidden_size

            # 解码起开始工作
            decoder_input = Variable(torch.LongTensor([[2]] * target_variable.size(0)))

            decoder_hidden = encoder_hidden

            # 同时采用两种方式解码
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                for di in range(max_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    # print(decoder_output.size())   # torch.Size([64, 40838])

                    # 计算损失
                    loss += criterion(decoder_output, target_variable[:, di])

                    decoder_input = target_variable[:, di].unsqueeze(1)  # batch_size x length_Seq
            else:
                for di in range(max_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

                    topv, topi = decoder_output.data.topk(1, dim=1)  # 返回两个值　分别是最大值，以及最大值的下标　额外强调一下　这是批量

                    ni = topi[:, 0]  # 相当于把那些最大值的下标拿出来作为下一步的输入
                    decoder_input = Variable(ni.unsqueeze(1))

                    # 计算损失
                    loss += criterion(decoder_output, target_variable[:, di])

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            # print_loss_total += loss.data.numpy()[0]
            print("当前epoch:{}, step:{}, loss:{}".format(epoch, k, loss))
