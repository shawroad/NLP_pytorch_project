"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-02-21
"""
import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size, num_hiddens, num_layers, dropout=drop_prob
        )

    def forward(self, inputs, state):
        # inputs.size()   # batch_size, max_len
        embedding = self.embedding(inputs.long()).permute(1, 0, 2)
        # max_len, batch_size, embed_dim
        output = self.rnn(embedding, state)
        # print(output[0].size())   # torch.Size([7, 2, 64])
        # print(output[1].size())   # torch.Size([2, 2, 64])
        return output

    def begin_state(self):
        # 可以自定义初始化
        return None


def attention_model(input_size, attention_size):
    model = nn.Sequential(
        nn.Linear(input_size, attention_size, bias=False),
        nn.Tanh(),
        nn.Linear(attention_size, 1, bias=False)
    )
    return model


def attentiom_forward(model, enc_states, dec_states):
    '''
    :param model:
    :param enc_states: max_len, batch_size, hidden_size
    :param dec_states: batch_size, hidden_size
    :return:
    '''
    # print(enc_states.size())   # torch.Size([7, 2, 64])
    # print(dec_states.size())   # torch.Size([2, 64])
    dec_states = dec_states.unsqueeze(dim=0).expand_as(enc_states)
    # print(dec_states.size())    # torch.Size([7, 2, 64])
    enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
    e = model(enc_and_dec_states)
    alpha = F.softmax(e, dim=0)
    return (alpha * enc_states).sum(dim=0)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 attention_size, drop_prob=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(2 * num_hiddens, attention_size)

        self.rnn = nn.GRU(num_hiddens + embed_size, num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, state, enc_states):
        '''
                dec_output, dec_state = decoder(
            dec_input, dec_state, enc_outputs
        )

        :param cur_input: batch_size,
        :param state: num_layers, batch_size, hidden_size
        :param enc_states: max_len, batch_size, hidden_size
        :return:
        '''
        c = attentiom_forward(self.attention, enc_states, state[-1])
        # print(c.size())    # torch.Size([2, 64])   # 这是注意力向量
        input_and_c = torch.cat((self.embedding(cur_input), c), dim=1)
        # print(input_and_c.size())    # torch.Size([2, 128])
        output, state = self.rnn(input_and_c.unsqueeze(0), state)
        output = self.out(output).squeeze(dim=0)
        return output, state

    def begin_state(self, enc_state):
        return enc_state


