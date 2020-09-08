"""

@file  : DocQA.py

@author: xiaolu

@time  : 2020-03-16

"""
"""

@file  : QANet.py

@author: xiaolu

@time  : 2020-01-20

"""
from torch import nn
import torch
import torch.nn.functional as F
import math
from config import Config
from torch.autograd import Variable


d_model = Config.d_model
n_head = Config.num_heads
d_word = Config.glove_dim
d_char = Config.char_dim
batch_size = Config.batch_size
dropout = Config.dropout
dropout_char = Config.dropout_char

d_k = d_model // n_head
d_cq = d_model * 4
len_c = Config.para_limit
len_q = Config.ques_limit


class DepthwiseSeparableConv(nn.Module):
    '''
    两种类型的卷积, 可以针对一维, 也可以针对二维
    '''
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class Highway(nn.Module):
    '''
    带门限机制的全连接
    '''
    def __init__(self, layer_num: int, size: int):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        x = x.transpose(1, 2)
        return x


class Embedding(nn.Module):
    '''
    词嵌入部分
    print(Qw.size())   # torch.Size([2, 50, 300])
    print(Qc.size())   # torch.Size([2, 50, 16, 64])
    '''
    def __init__(self):
        super().__init__()
        self.conv2d = DepthwiseSeparableConv(d_char, d_char, 5, dim=2)
        self.high = Highway(2, d_word+d_char)

    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)   # [2, 400, 16, 64]
        # print(ch_emb.size())  # torch.Size([2, 64, 400, 16])
        # 解读上述维度 400代表行 也就是有400个词, 16代表列, 也就是每个词的长度.  64可以想象成通道 咱们刚才是将每个字符都嵌入成了64维度

        ch_emb = F.dropout(ch_emb, p=Config.dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        # print(ch_emb.size())   # torch.Size([2, 64, 400, 16]) 把这些字符信息经过卷积揉搓了一下

        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        # print(ch_emb.size())   # torch.Size([2, 64, 400]) 相当于最大化池化 每个字符调出维度中最大的那个值

        ch_emb = ch_emb.squeeze()
        # print(ch_emb.size())  # torch.Size([2, 64, 400])

        wd_emb = F.dropout(wd_emb, p=Config.dropout, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        # print(wd_emb.size())   # torch.Size([2, 300, 400])

        emb = torch.cat([ch_emb, wd_emb], dim=1)
        # print(emb.size())   # torch.Size([2, 364, 400])

        emb = self.high(emb)
        # print(emb.size())   # torch.Size([2, 364, 400])
        return emb


class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0, dropout_output=False,
                 rnn_type=nn.LSTM, concat_layers=False, padding=False):

        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort).to(Config.device)
        idx_unsort = Variable(idx_unsort).to(Config.device)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type()).to(Config.device)
            output = torch.cat([output, Variable(padding).to(Config.device)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class BiAttention(nn.Module):
    """
        biattention in BiDAF model
        :param dim: hidden size
    """
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(3 * dim, 1)

    def forward(self, x1, x1_mask, x2, x2_mask):
        """
        :param x1: b x n x d
        :param x2: b x m x d
        :param x1_mask: b x n
        :param x2_mask: b x m
        """
        # bxnxmxd
        x1_aug = x1.unsqueeze(2).expand(x1.size(0), x1.size(1), x2.size(1), x1.size(2))
        x2_aug = x2.unsqueeze(1).expand(x1.size(0), x1.size(1), x2.size(1), x2.size(2))
        x_input = torch.cat([x1_aug, x2_aug, x1_aug * x2_aug], dim=3)
        similarity = self.linear(x_input).squeeze(3)
        # bxnxm
        x2_mask = x2_mask.unsqueeze(1).expand_as(similarity)
        similarity.data.masked_fill_(x2_mask.data, -2e20)
        # bxnxm
        # c -> q
        sim_row = F.softmax(similarity, dim=2)
        attn_a = sim_row.bmm(x2)
        # q -> c
        x1_mask = x1_mask.unsqueeze(2).expand_as(similarity)
        similarity.data.masked_fill_(x1_mask.data, -2e20)
        sim_col = F.softmax(similarity, dim=1)
        q2c = sim_col.transpose(1,2).bmm(x1)
        attn_b = sim_row.bmm(q2c)
        return attn_a, attn_b


class DocQA(nn.Module):

    def __init__(self, word_mat, char_mat):
        super().__init__()
        # 加载所谓的词向量, 字符向量
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=Config.pretrained_char)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))

        # 将词向量和字符向量进行揉搓
        self.emb = Embedding()

        self.enc_rnn = StackedBRNN(
            input_size=364,
            hidden_size=64,
            num_layers=2,
            dropout_rate=0.2,
            dropout_output=True,
            concat_layers=True,
            rnn_type=nn.GRU,
            padding=True,
        )

        self.qc_attn = BiAttention(64*4)

        self.linear_attn = nn.Linear(64 * 4 * 4, 64*4)

        self.relu_attn = nn.ReLU()

        self.self_rnn = StackedBRNN(
                    input_size=64*4,
                    hidden_size=64,
                    num_layers=2,
                    dropout_rate=0.2,
                    dropout_output=True,
                    concat_layers=True,
                    rnn_type=nn.GRU,
                    padding=True,
                )

        self.cc_attn = BiAttention(64*4)

        self.linear_self = nn.Linear(64 * 4 * 4, 64*4)

        self.relu_self = nn.ReLU()

        self.fusion_rnn1 = StackedBRNN(
                    input_size=64*4,
                    hidden_size=64,
                    num_layers=2,
                    dropout_rate=0.2,
                    dropout_output=True,
                    concat_layers=True,
                    rnn_type=nn.GRU,
                    padding=True,
                )

        self.start = nn.Sequential(
                    nn.Linear(64*4, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )

        self.fusion_rnn2 = StackedBRNN(
                    input_size=64 * 4 * 2,
                    hidden_size=64,
                    num_layers=2,
                    dropout_rate=0.2,
                    dropout_output=True,
                    concat_layers=True,
                    rnn_type=nn.GRU,
                    padding=True,
                )

        self.end = nn.Sequential(
                nn.Linear(64*4, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
                )

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        # 对问题和文章进行mask
        cmask = (torch.zeros_like(Cwid) == Cwid)
        qmask = (torch.zeros_like(Qwid) == Qwid)
        # print(cmask)   # 把padding的部分全部置为1, 把真实存在数据的部分置为0
        # print(qmask)

        # 对文章和问题进行词嵌入和字符嵌入
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        # print(Cw.size())   # torch.Size([2, 400, 300])   batch_size x context_max_len x hidden_size
        # print(Cc.size())  # torch.Size([2, 400, 16, 64]) (batch_size,context_max_len,word_max_len,word_embedding_size)

        # 问题
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        # print(Qw.size())   # torch.Size([2, 50, 300])
        # print(Qc.size())   # torch.Size([2, 50, 16, 64])

        # 将文章的词向量,字符向量进行融合　　　将问题的词向量,字符向量进行融合
        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        # print(C.size())   # torch.Size([2, 364, 400])
        # print(Q.size())   # torch.Size([2, 364, 50])

        C = C.permute((0, 2, 1))
        Q = Q.permute((0, 2, 1))

        x1_pro = self.enc_rnn(C, cmask)
        x2_pro = self.enc_rnn(Q, qmask)
        # print(x1_pro.size())   # torch.Size([2, 400, 256])
        # print(x2_pro.size())   # torch.Size([2, 50, 256])

        attn_A, attn_B = self.qc_attn(x1_pro, cmask, x2_pro, qmask)
        # print(attn_A.size())   # torch.Size([2, 400, 256])
        # print(attn_B.size())   # torch.Size([2, 400, 256])

        c_fusion = torch.cat([x1_pro, attn_A, x1_pro * attn_A, attn_A * attn_B], dim=2)   # , attn_A * attn_B

        # print(c_fusion.size())   # torch.Size([2, 400, 1024])
        c_fusion = self.linear_attn(c_fusion)
        c_fusion = self.relu_attn(c_fusion)
        # print(c_fusion.size())   # torch.Size([2, 400, 256])

        self_attn_input = self.self_rnn(c_fusion, cmask)
        # print(self_attn_input.size())   # torch.Size([2, 400, 256])

        self_attn_A, self_attn_B = self.cc_attn(self_attn_input, cmask,
                                                self_attn_input, cmask)
        # print(self_attn_A.size())   # torch.Size([2, 400, 256])
        # print(self_attn_B.size())   # torch.Size([2, 400, 256])

        self_fusion = torch.cat([self_attn_input,
                                 self_attn_A,
                                 self_attn_input * self_attn_A,
                                 self_attn_A * self_attn_B],
                                dim=2)
        # print(self_fusion.size())  # torch.Size([2, 400, 1024])

        self_fusion = self.relu_attn(self.linear_self(self_fusion))

        context = self_fusion + c_fusion  # 256
        # print(context.size())   # torch.Size([2, 400, 256])

        start_context = self.fusion_rnn1(context, cmask)
        # print(start_context.size())   # torch.Size([2, 400, 256])

        start_logits = self.start(start_context).squeeze(2)

        start_logits.data.masked_fill_(cmask.data, -2e20)

        softmax_start = F.softmax(start_logits, dim=1)

        re_fusion = torch.cat([context, start_context], dim=2)

        g2 = self.fusion_rnn2(re_fusion, cmask)

        end_logits = self.end(g2).squeeze(2)

        end_logits.data.masked_fill_(cmask.data, -2e20)

        softmax_end = F.softmax(end_logits, dim=1)

        # if self.training:
        #     start_scores = torch.log(softmax_start + 1e-20)
        #     end_scores = torch.log(softmax_end + 1e-20)
        # else:
        #     start_scores = softmax_start
        #     end_scores = softmax_end

        start_scores = torch.log(softmax_start + 1e-20)
        end_scores = torch.log(softmax_end + 1e-20)
        return start_scores, end_scores



