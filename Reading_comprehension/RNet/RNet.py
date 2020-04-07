"""

@file  : RNet.py

@author: xiaolu

@time  : 2020-03-09

"""
import torch
from torch import nn
import torch.nn.functional as F
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
        # torch.Size([2, 400, 16, 64])

        ch_emb = ch_emb.permute(0, 3, 1, 2)   # [2, 400, 16, 64]
        # print(ch_emb.size())  # torch.Size([2, 64, 400, 16])

        # 解读上述维度 400代表行 也就是有400个词, 16代表列, 也就是每个词的长度.  64可以想象成通道 咱们刚才是将每个字符都嵌入成了64维度
        ch_emb = F.dropout(ch_emb, p=Config.dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb)   # 相当于把字符嵌入的维度当做通道　进行三维卷积
        # print(ch_emb.size())   # torch.Size([2, 64, 400, 16]) 把这些字符信息经过卷积揉搓了一下

        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)   # 从第三维度中选取最大　相当于从单词中选出关键字符
        # print(ch_emb.size())   # torch.Size([2, 64, 400]) 相当于最大化池化 每个字符调出维度中最大的那个值

        ch_emb = ch_emb.squeeze()
        # print(ch_emb.size())  # torch.Size([2, 64, 400])

        wd_emb = F.dropout(wd_emb, p=Config.dropout, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        # print(wd_emb.size())   # torch.Size([2, 300, 400])

        emb = torch.cat([ch_emb, wd_emb], dim=1)
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


class DotAttention(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, hidden):
        super(DotAttention, self).__init__()
        self.hidden = hidden
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, hidden)
        self.linear2 = nn.Linear(input_size, hidden)
        self.linear3 = nn.Linear(2*input_size, 2*input_size)

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        #x_proj = F.relu(self.linear1(x))
        #y_proj = F.relu(self.linear2(y))
        x_proj = x
        y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1)) / (self.hidden ** 0.5)

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        res = torch.cat([x, matched_seq], dim=2)
        res = F.dropout(res, p=0.2, training=self.training)

        # add gate
        gate = F.sigmoid(self.linear3(res))

        return res * gate


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        scores = self.linear2(F.tanh(self.linear1(x))).squeeze(2)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=-1)
        res = alpha.unsqueeze(1).bmm(x).squeeze(1)
        return res


class PtrNet(nn.Module):
    def __init__(self, in_size):
        super(PtrNet, self).__init__()
        self.linear1 = nn.Linear(in_size, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, doc, q_vec, x1_mask):
        """
        :param p: B * N * H
        :param q_vec: B * H
        :param x1_mask: B * N
        :return res: B * H
        :return out: B * N
        """
        out = torch.cat([doc, q_vec.unsqueeze(1).expand(q_vec.size(0), doc.size(1), q_vec.size(1))], dim=2)
        out = F.tanh(self.linear1(out))
        out = self.linear2(out).squeeze(2) # B * N
        out.data.masked_fill_(x1_mask.data, -float('inf'))
        out = F.softmax(out, dim=-1)
        res = out.unsqueeze(1).bmm(doc).squeeze(1) # b*h

        return res, out


class RNet(nn.Module):
    def __init__(self, word_mat, char_mat):
        super().__init__()
        # # 加载所谓的词向量, 字符向量
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=Config.pretrained_char)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))

        # 将词向量和字符向量进行揉搓
        self.emb = Embedding()

        self.encode_rnn = StackedBRNN(
            input_size=364,
            hidden_size=64,
            num_layers=3,
            dropout_rate=0.2,
            dropout_output=False,
            rnn_type=nn.GRU,
            padding=True,
        )

        outdim = 2 * 64

        self.docattn = DotAttention(outdim, 128)

        self.attn_rnn1 = StackedBRNN(
            input_size=outdim * 2,
            hidden_size=128,
            num_layers=1,
            dropout_rate=0,
            dropout_output=True,
            concat_layers=False,
            rnn_type=nn.GRU,
            padding=True,
        )

        self.selfattn = DotAttention(128*2, 128)

        self.attn_rnn2 = StackedBRNN(
            input_size=128 * 2 * 2,
            hidden_size=128,
            num_layers=1,
            dropout_rate=0,
            dropout_output=False,
            concat_layers=False,
            rnn_type=nn.GRU,
            padding=True,
        )

        self.q_attn = LinearSeqAttn(2 * 64)

        # pointer network
        self.start_ptr = PtrNet(2 * 64 + 128 * 2)
        self.end_ptr = PtrNet(2 * 64 + 128 * 2)

        self.ptr_rnn = nn.GRUCell(256, 128)

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        # 对问题和文章进行mask
        # cmask = (torch.zeros_like(Cwid) == Cwid).float()
        # qmask = (torch.zeros_like(Qwid) == Qwid).float()
        cmask = (torch.zeros_like(Cwid) == Cwid)
        qmask = (torch.zeros_like(Qwid) == Qwid)

        # print(cmask)   # 把padding的部分全部置为1, 把真实存在数据的部分置为0
        # print(cmask.size())   # torch.Size([2, 400])
        # print(Cwid.size())   # torch.Size([2, 400])
        # print(Ccid.size())   # torch.Size([2, 400, 16])

        # 文章
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        # print(Cw.size())   # torch.Size([2, 400, 300])
        # print(Cc.size())   # torch.Size([2, 400, 16, 64])

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

        C = self.encode_rnn(C, cmask)
        Q = self.encode_rnn(Q, qmask)
        # print(C.size())   # torch.Size([2, 400, 128])
        # print(Q.size())   # torch.Size([2, 50, 128])

        qc_att = self.docattn(C, Q, qmask)
        # print(qc_att.size())  # torch.Size([2, 400, 256])

        qc_att = self.attn_rnn1(qc_att, cmask)
        # print(qc_att.size())   # torch.Size([2, 400, 256])

        self_att = self.selfattn(qc_att, qc_att, cmask)
        # print(self_att.size())   # torch.Size([2, 400, 512])

        match = self.attn_rnn2(self_att, cmask)
        # print(match.size())   # torch.Size([2, 400, 256])

        # self attention convert question to q_vector
        q_vec = self.q_attn(Q, qmask)
        # print(q_vec.size())   # torch.Size([2, 128])

        # 指针网络预测
        internal, start_scores = self.start_ptr(match, q_vec, cmask)
        internal = self.ptr_rnn(internal, q_vec)
        _, end_scores = self.end_ptr(match, internal, cmask)

        if self.training:
            start_scores = torch.log(start_scores + 1e-10)
            end_scores = torch.log(end_scores + 1e-10)
        return start_scores, end_scores
