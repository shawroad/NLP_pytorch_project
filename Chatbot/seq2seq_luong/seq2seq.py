"""

@file  : seq2seq.py

@author: xiaolu

@time  : 2020-04-01

"""
from torch import nn
import torch
import torch.nn.functional as F
from config import Config


# 定义标志位置
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class EncoderRNN(nn.Module):
    # 编码部分
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        # print(input_seq.size())  # torch.Size([9, 5])
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # print(outputs)   # outputs是每步的输出　padding部分是全零向量
        # print(_)   # 返回的是序列长度　即每个句子未padding的真实长度
        # print(outputs.size())   # torch.Size([8, 5, 1000])  seq_len x batch_size x (bidirection x hidden_size)

        # 将两个方向的输出进行相加
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # print(outputs.size())   # torch.Size([9, 5, 500])  seq_len x batch_size x hidden_size
        # print(hidden.size())   # torch.Size([4, 5, 500])   (2layers x 2bidirection) x batch_size x hidden_size
        return outputs, hidden


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


class LuongAttnDecoderRNN(nn.Module):
    # 解码部分
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # 每次执行一步
        embedded = self.embedding(input_step)   # 对当前输入一步进行词嵌入
        embedded = self.embedding_dropout(embedded)   # 词嵌入后加入dropout
        # print(embedded.size())   # torch.Size([1, 5, 500])  seq_len x batch_size x embedding_size

        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)  # last_hidden是编码最后的隐态裁剪出来的
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


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        '''
        searcher(input_batch, lengths, max_length)
        :param input_seq: # torch.Size([7, 1])
        :param input_length: # [7, ]
        :param max_length: 10
        :return:
        '''

        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # print(encoder_outputs.size())   # torch.Size([7, 1, 500])  seq_len x batch_size x hidden_size
        # print(encoder_hidden.size())  # torch.Size([4, 1, 500])     4(两层 双向) x batch_size x hidden_size

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token

        # 构造起始标志
        decoder_input = torch.ones(1, 1, device=Config.device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to

        all_tokens = torch.zeros([0], device=Config.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=Config.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # print(decoder_output.size())   # torch.Size([1, 7826])

            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # decoder_scores当前步 最高概率   decoder_input 最高概率对应的词

            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
