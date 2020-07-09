# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 9:26
# @Author  : xiaolu
# @FileName: model.py
# @Software: PyCharm
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils import rnn
from config import Config


class Model(nn.Module):
    def __init__(self, word_mat, char_mat):
        super(Model, self).__init__()
        self.word_dim = Config.glove_dim
        self.char_hidden = Config.char_hidden
        self.hidden = Config.hidden

        # 词嵌入
        self.word_emb = nn.Embedding(len(word_mat), len(word_mat[0]), padding_idx=0)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_mat))
        self.word_emb.weight.requires_grad = False   # 不进行方向传播

        # 字潜入
        self.char_emb = nn.Embedding(len(char_mat), len(char_mat[0]), padding_idx=0)
        self.char_emb.weight.data.copy_(torch.from_numpy(char_mat))

        self.char_cnn = nn.Conv1d(Config.char_dim, Config.char_hidden, 5)

        # rnn
        self.rnn = EncoderRNN(Config.char_hidden + self.word_dim, Config.hidden, 1, True, True, 1 - Config.keep_prob, False)

        self.qc_att = BiAttention(Config.hidden * 2, 1 - Config.keep_prob)

        self.linear_1 = nn.Sequential(
                nn.Linear(Config.hidden*8, Config.hidden),
                nn.ReLU()
            )

        self.rnn_2 = EncoderRNN(Config.hidden, Config.hidden, 1, False, True, 1 - Config.keep_prob, False)

        self.self_att = BiAttention(Config.hidden * 2, 1 - Config.keep_prob)
        self.linear_2 = nn.Sequential(
            nn.Linear(Config.hidden * 8, Config.hidden),
            nn.ReLU()
        )

        self.rnn_sp = EncoderRNN(Config.hidden, Config.hidden, 1, False, True, 1 - Config.keep_prob, False)
        self.linear_sp = nn.Linear(Config.hidden * 2, 1)

        self.rnn_start = EncoderRNN(Config.hidden + 1, Config.hidden, 1, False, True, 1 - Config.keep_prob, False)
        self.linear_start = nn.Linear(Config.hidden * 2, 1)

        self.rnn_end = EncoderRNN(Config.hidden * 3 + 1, Config.hidden, 1, False, True, 1 - Config.keep_prob, False)
        self.linear_end = nn.Linear(Config.hidden * 2, 1)

        self.rnn_type = EncoderRNN(Config.hidden * 3 + 1, Config.hidden, 1, False, True, 1 - Config.keep_prob, False)
        self.linear_type = nn.Linear(Config.hidden * 2, 3)

        self.cache_S = 0

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs,
                context_lens, start_mapping, end_mapping, all_mapping, return_yp=False):
        para_size, ques_size, char_size, bsz = context_idxs.size(1), ques_idxs.size(1), context_char_idxs.size(2), context_idxs.size(0)

        context_mask = (context_idxs > 0).float()   # 零位置认为是padding
        ques_mask = (ques_idxs > 0).float()

        # 1. 对于字
        # print(self.char_emb(context_char_idxs.contiguous()).size())    # torch.Size([2, 917, 16, 8])

        context_ch = self.char_emb(context_char_idxs.contiguous().view(-1, char_size)).view(bsz * para_size, char_size, -1)
        ques_ch = self.char_emb(ques_char_idxs.contiguous().view(-1, char_size)).view(bsz * ques_size, char_size, -1)
        # print(context_ch.size())   # torch.Size([1834, 16, 8])
        # 1834 8 16
        context_ch = self.char_cnn(context_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, para_size, -1)
        ques_ch = self.char_cnn(ques_ch.permute(0, 2, 1).contiguous()).max(dim=-1)[0].view(bsz, ques_size, -1)

        # 2. 对于词
        context_word = self.word_emb(context_idxs)
        ques_word = self.word_emb(ques_idxs)

        # 将字和词进行合并
        context_output = torch.cat([context_word, context_ch], dim=2)
        ques_output = torch.cat([ques_word, ques_ch], dim=2)
        # print(context_output.size())    # torch.Size([2, 917, 400])
        # print(ques_output.size())   # torch.Size([2, 18, 400])

        context_output = self.rnn(context_output, context_lens)
        ques_output = self.rnn(ques_output)
        # print(context_output.size())    # torch.Size([2, 917, 160])
        # print(ques_output.size())    # torch.Size([2, 18, 160])

        output = self.qc_att(context_output, ques_output, ques_mask)
        # print(output.size())    # torch.Size([2, 917, 640])

        output = self.linear_1(output)
        # print(output.size())    # torch.Size([2, 917, 80])

        output_t = self.rnn_2(output, context_lens)
        # print(output_t.size())   # torch.Size([2, 733, 160])

        output_t = self.self_att(output_t, output_t, context_mask)
        output_t = self.linear_2(output_t)

        output = output + output_t

        sp_output = self.rnn_sp(output, context_lens)

        start_output = torch.matmul(start_mapping.permute(0, 2, 1).contiguous(), sp_output[:, :, self.hidden:])
        end_output = torch.matmul(end_mapping.permute(0, 2, 1).contiguous(), sp_output[:, :, :self.hidden])
        sp_output = torch.cat([start_output, end_output], dim=-1)
        sp_output = self.linear_sp(sp_output)
        sp_output_aux = Variable(sp_output.data.new(sp_output.size(0), sp_output.size(1), 1).zero_())
        predict_support = torch.cat([sp_output_aux, sp_output], dim=-1).contiguous()

        sp_output = torch.matmul(all_mapping, sp_output)
        output = torch.cat([output, sp_output], dim=-1)

        output_start = self.rnn_start(output, context_lens)
        logit1 = self.linear_start(output_start).squeeze(2) - 1e30 * (1 - context_mask)
        output_end = torch.cat([output, output_start], dim=2)
        output_end = self.rnn_end(output_end, context_lens)
        logit2 = self.linear_end(output_end).squeeze(2) - 1e30 * (1 - context_mask)

        output_type = torch.cat([output, output_end], dim=2)
        output_type = torch.max(self.rnn_type(output_type, context_lens), 1)[0]
        predict_type = self.linear_type(output_type)
        # print(predict_type.size())    # torch.Size([2, 3])
        if not return_yp:
            return logit1, logit2, predict_type, predict_support

        outer = logit1[:, :, None] + logit2[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return logit1, logit2, predict_type, predict_support, yp1, yp2


class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super(EncoderRNN, self).__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))
