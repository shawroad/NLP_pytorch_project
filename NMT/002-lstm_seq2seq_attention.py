"""

@file  : 002-lstm_seq2seq_attention.py

@author: xiaolu

@time  : 2019-12-25

"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from torch import optim
import random


class Encoder_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_layers, hidden_size):
        super(Encoder_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=self.n_layers, batch_first=True, bidirectional=False)

    def forward(self, x, hidden):
        '''
        :param x: 输入的序列
        :param hidden: 初始化的隐层信息
        :return:
        '''
        x = self.embed(x)
        # print(x.size())   # torch.Size([30, 10, 128])  (batch_size, seq_len, embedding_dim)
        x, h = self.lstm(x)
        # print(x.size())     # torch.Size([30, 10, 256])
        # print(h[0].size())   # torch.Size([4, 30, 128])
        # print(h[1].size())   # torch.Size([4, 30, 128])
        return x, h

    def initHidden(self, batch_size):
        '''
        对隐藏单元进行初始化
        :param batch_size:
        :return:
        '''
        h = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        c = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        hidden = [h, c]
        return hidden


class Decoder_model(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Decoder_model, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, hidden_size)

        self.Linear = nn.Linear(2*hidden_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False)

        self.logits = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, encoder_output):
        x = self.embed(x)
        # print(x.size())  # torch.Size([30, 1, 128])

        # 这里取出hidden[1] 也就是取出细胞状态c
        c = torch.transpose(hidden[1], 0, 1).contiguous()   # torch.Size([30, 4, 128])
        c = c.view(c.size(0), -1)
        c = c.unsqueeze(2)

        atten = encoder_output.bmm(c)
        atten = atten.view(atten.size(0), -1)
        atten_weight = F.softmax(atten, -1)
        # print(atten_weight.size())  # torch.Size([64, 15])

        # 接下来输出编码向量乘权重
        atten_weight = atten_weight.unsqueeze(1)
        atten_appiled = torch.bmm(atten_weight, encoder_output)  # batch_size x 1 x hidden_size * direction
        inputs = torch.cat((atten_appiled, x), -1)
        # print(inputs.size())  # 这个将注意力向量和embedding向量进行合并

        # 接下来通过一个全连接将其维度搞成我们的输入维度
        inputs = self.Linear(inputs)

        output, hidden = self.lstm(inputs)

        output = output.view(output.size(0), -1)
        output = self.logits(output)

        return output, hidden, atten_appiled


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
    embedding_size = 32
    n_layers = 1
    eng_vocab_size = len(eng_vocab2id)
    fra_vocab_size = len(fra_vocab2id)

    encoder = Encoder_model(eng_vocab_size, embedding_size, n_layers, hidden_size)
    decoder = Decoder_model(fra_vocab_size, hidden_size)

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
            encoder_output, encoder_hidden = encoder(input_variable, encoder_hidden)

            # 解码起开始工作
            decoder_input = Variable(torch.LongTensor([[2]] * target_variable.size(0)))

            decoder_hidden = encoder_hidden

            # 同时采用两种方式解码
            # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            use_teacher_forcing = True

            if use_teacher_forcing:
                for di in range(max_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
                    # print(decoder_output.size())   # torch.Size([64, 40838])
                    # print(decoder_hidden[0].size())    # torch.Size([1, 64, 32])
                    # print(decoder_attention.size())

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

        if epoch % 10 == 0:
            indices = np.random.choice(range(len(eng_id_data)), 20)
            for ind in indices:
                data = eng_id_data[ind]
                target = fra_id_data[ind]
                input_variable = Variable(torch.LongTensor(data))
                target_variable = Variable(torch.LongTensor(target))
                input_variable = input_variable.unsqueeze(0)
                target_variable = target_variable.unsqueeze(0)

                encoder_hidden = encoder.initHidden(input_variable.size(0))

                loss = 0
                encoder_output, encoder_hidden = encoder(input_variable, encoder_hidden)

                decoder_input = Variable(torch.LongTensor([[2]] * target_variable.size(0)))

                decoder_hidden = encoder_hidden
                output_sentence = []

                rights = []
                for di in range(max_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
                    topv, topi = decoder_output.data.topk(1, dim=1)
                    ni = topi[:, 0]
                    decoder_input = Variable(ni.unsqueeze(1))
                    ni = ni.numpy()[0]
                    output_sentence.append(ni)

                sentence = [fra_id2vocab.get(i) for i in output_sentence]
                print("英文:", ' '.join([eng_id2vocab.get(i) for i in data]))
                print("人工翻译:", ' '.join([fra_id2vocab.get(i) for i in target]))
                print("机器翻译:", ' '.join(sentence))








