"""

@file  : model.py

@author: xiaolu

@time  : 2020-04-15

"""
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import torch
from config import Config


class MultiHeadAttention(nn.Module):
    '''
    Multi-Head Attention module
    '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # batch_size x max_len x n_heads x q_dim
        # print("q维度:", q.size())  # q维度: torch.Size([2, 19, 8, 64])

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled Dot-Product Attention
    '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # print(q.size())   # torch.Size([16, 19, 64])
        # print(k.size())   # torch.Size([16, 19, 64])

        attn = torch.bmm(q, k.transpose(1, 2))
        # print(attn.size())

        attn = attn / self.temperature
        # print(attn.size())  # torch.Size([16, 19, 19])

        if mask is not None:
            attn = attn.masked_fill(mask.bool(), -np.inf)  # 这里相当于给置成1的那一块填充成-inf  加下来算softmax的时候， 概率基本就是零了

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class PositionalEncoding(nn.Module):
    """Implement the positional encoding (PE) function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        Args:
            input: N x T x D
        """
        length = input.size(1)
        return self.pe[:, :length]


class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """
    padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: batch_size x max_len
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)


def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    '''
    mask position is set to 1
    :param padded_input: batch_size x max_len
    :param input_lengths: batch_size
    :param expand_length:
    :return:
    '''
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)   # 相当于把是零的全部置为True 其余位置置为False
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask


class EncoderLayer(nn.Module):
    """
    Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, embedding, n_src_vocab=Config.vocab_size, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_model=512, d_inner=2048, dropout=0.1, pe_maxlen=Config.max_len):
        '''
        :param n_src_vocab: 输入文本的词表大小
        :param n_layers: 基层编码块
        :param n_head: self_attention需要几个头
        :param d_k: q, k 查询向量和键向量
        :param d_v: value维度
        :param d_model: 隐层维度
        :param d_inner: 前馈网络的中间那一步的维度
        :param dropout:
        :param pe_maxlen: 这个应该算的是位置向量吧
        '''
        super(Encoder, self).__init__()
        # parameters
        self.n_src_vocab = n_src_vocab
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen

        self.src_emb = embedding
        # self.src_emb = nn.Embedding(n_src_vocab, d_model, padding_idx=Config.PAD)
        self.pos_emb = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, 1)

    def forward(self, padded_input, input_lengths, return_attns=False):
        """
        Args:
            padded_input: batch_size x max_len
            input_lengths: batch_size
        Returns:
            enc_output: batch_size x max_len x hidden_size
        """
        enc_slf_attn_list = []

        # Forward
        enc_outputs = self.src_emb(padded_input)
        enc_outputs += self.pos_emb(enc_outputs)
        enc_output = self.dropout(enc_outputs)

        # Prepare masks
        non_pad_mask = get_non_pad_mask(enc_output, input_lengths=input_lengths)  # 将填充的位置padding为0
        length = padded_input.size(1)
        # print(length)   # 19 也就是第一批数据中输入最长的长度为19

        slf_attn_mask = get_attn_pad_mask(enc_output, input_lengths, length)
        # print(slf_attn_mask.size())  # torch.Size([2, 19, 19]) batch_size x max_len x max_len
        # print(slf_attn_mask)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list

        # 这里将全部输出通过一定的手段揉搓成一个向量 为了解码的隐层
        prob = self.linear(enc_output)
        prob = prob.squeeze()
        prob = F.softmax(prob, dim=1)
        prob = prob.unsqueeze(1)

        # print(prob.size())   # torch.Size([2, 1, 492])
        hidden = torch.bmm(prob, enc_output)
        hidden = hidden.squeeze()


        return enc_output, hidden


class Decoder(nn.Module):
    # 解码器
    def __init__(self, attn_model, embedding, n_layers=1, dropout=0.1, hidden_size=512):
        super(Decoder, self).__init__()
        self.attn_model = attn_model
        self.attn = Attn(attn_model, hidden_size)

        self.n_layers = n_layers
        self.dropout = dropout
        self.output_size = Config.vocab_size
        # Define layers
        self.embedding = embedding

        self.embedding_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, self.output_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # 每次执行一步
        embedded = self.embedding(input_step)   # 对当前输入一步进行词嵌入
        embedded = self.embedding_dropout(embedded)   # 词嵌入后加入dropout
        # print(embedded.size())   # torch.Size([1, 5, 500])  seq_len x batch_size x embedding_size

        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded)  # last_hidden是编码最后的隐态裁剪出来的
        # print(rnn_output.size())   # torch.Size([1, 5, 500])
        # print(hidden.size())   # torch.Size([2, 5, 500])   # 这里第一维度是2 是因为解码用了两层

        # 计算当前步注意力权重
        attn_weights = self.attn(rnn_output, encoder_outputs)  # torch.Size([5, 1, 10]) batch_size x 1 x seq_length

        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        # encoder_outputs  size=torch.Size([10, 5, 500])  -> size(5, 10, 500)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (5, 1, 10) x (5, 10, 500)

        # print(context.size())   # torch.Size([5, 1, 500])  # 将编码的输出整合成一个向量 是对当前步影响的注意力向量

        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)  # torch.Size([1, 5, 500]) -> torch.Size([5, 500])

        context = context.squeeze(1)    # torch.Size([5, 1, 500]) -> torch.Size([5, 500])

        concat_input = torch.cat((rnn_output, context), 1)   # 注意力向量和当前步的输出向量进行拼接
        # print(concat_input.size())    # torch.Size([5, 1000])  (batch_size, 2 * hidden_size)

        concat_output = torch.tanh(self.concat(concat_input))  # self.concat() 是将2 * hidden_size 压成 hidden_size
        # print(concat_output.size())   # torch.Size([5, 500])

        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


class Attn(nn.Module):
    # Luong attention layer
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: size (1, 5, 500)  # 解码当前步的输出
        :param encoder_outputs: size (10, 5, 500)  # 编码每一步的输出
        :return:
        '''
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
            # torch.sum((1, 5, 500) * (10, 5, 500), dim=2)
            # 解读: 相当于用这一步的向量去乘编码每步输出的向量 然后得到的维度还是(10, 5, 500) 接下来再把每步的向量进行累加
            # print(attn_energies.size())   # torch.Size([10, 5])
            # print(attn_energies)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        # print(attn_energies.size())   # torch.Size([5, 10])

        # Return the softmax normalized probability scores (with added dimension)
        # out = F.softmax(attn_energies, dim=1).unsqueeze(1)
        # print(out.size())  # torch.Size([5, 1, 10])
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

