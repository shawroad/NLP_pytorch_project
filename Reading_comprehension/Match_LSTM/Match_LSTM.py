"""

@file  : Match_LSTM.py

@author: xiaolu

@time  : 2020-03-10

"""
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import Config


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


def mask_logits(target, mask):
    return target * (1-mask) + mask * (-1e30)


class MultiHeadAttention(nn.Module):
    '''
    多头注意力
    '''
    def __init__(self):
        super().__init__()

        self.q_linear = nn.Linear(Config.d_model, Config.d_model)
        self.v_linear = nn.Linear(Config.d_model, Config.d_model)
        self.k_linear = nn.Linear(Config.d_model, Config.d_model)
        self.dropout = nn.Dropout(Config.dropout)
        self.fc = nn.Linear(Config.d_model, Config.d_model)
        self.a = 1 / math.sqrt(d_k)

    def forward(self, x, mask):
        bs, _, l_x = x.size()
        x = x.transpose(1, 2)
        k = self.k_linear(x).view(bs, l_x, Config.num_heads, d_k)
        q = self.q_linear(x).view(bs, l_x, Config.num_heads, d_k)
        v = self.v_linear(x).view(bs, l_x, Config.num_heads, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(bs * Config.num_heads, l_x, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(bs * Config.num_heads, l_x, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(bs * Config.num_heads, l_x, d_k)
        mask = mask.unsqueeze(1).expand(-1, l_x, -1).repeat(Config.num_heads, 1, 1)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.a
        attn = mask_logits(attn, mask)
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)
        out = out.view(Config.num_heads, bs, l_x, d_k).permute(1, 2, 0, 3).contiguous().view(bs, l_x, Config.d_model)
        out = self.fc(out)
        out = self.dropout(out)
        return out.transpose(1, 2)


class PosEncoder(nn.Module):
    '''
    位置编码
    '''
    def __init__(self, length):
        super().__init__()
        freqs = torch.Tensor([10000 ** (-i / Config.d_model) if i % 2 == 0 else -10000 ** ((1 - i) / Config.d_model) for i in range(Config.d_model)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(Config.d_model)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(Config.d_model, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x):
        x = x + self.pos_encoding
        return x


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, length: int):
        '''
        :param conv_num: 来多少次卷积
        :param ch_num: 线性输出维度
        :param k: 每次卷积时 卷积核的个数
        :param length: 文本长度(padding过后)
        '''
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])

        self.self_att = MultiHeadAttention()

        self.fc = nn.Linear(ch_num, ch_num, bias=True)

        self.pos = PosEncoder(length)
        # self.norm = nn.LayerNorm([d_model, length])
        self.normb = nn.LayerNorm([Config.d_model, length])

        self.norms = nn.ModuleList([nn.LayerNorm([Config.d_model, length]) for _ in range(conv_num)])
        self.norme = nn.LayerNorm([Config.d_model, length])
        self.L = conv_num

    def forward(self, x, mask):
        out = self.pos(x)   # 输入向量与位置向量的混合
        # print(out.size())  # torch.Size([2, 96, 400])

        res = out
        out = self.normb(out)
        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = Config.dropout * (i + 1) / self.L
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out)
        # print("Before attention: {}".format(out.size()))  # [2, 96, 400]
        out = self.self_att(out, mask)
        # print("After attention: {}".format(out.size()))  # [2, 96, 400]

        out = out + res
        out = F.dropout(out, p=Config.dropout, training=self.training)
        res = out
        out = self.norme(out)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        # print(out.size())  # torch.Size([2, 96, 400])
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=Config.dropout, training=self.training)

        return out


class Pointer(nn.Module):
    '''
    指针网络结构
    '''
    def __init__(self):
        super().__init__()
        w1 = torch.empty(Config.d_model * 2)
        w2 = torch.empty(Config.d_model * 2)
        lim = 3 / (2 * Config.d_model)
        nn.init.uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
        nn.init.uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, M1, M2, M3, mask):
        '''
        :param M1: torch.Size([2, 96, 400])
        :param M2: torch.Size([2, 96, 400])
        :param M3: torch.Size([2, 96, 400])
        :param mask:
        :return:
        '''
        # M1和M2搭配解出起始标志　　M1和M3搭配解出结束标志
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        # print(X1.size())   # torch.Size([2, 192, 400])
        # print(X2.size())   # torch.Size([2, 192, 400])

        Y1 = torch.matmul(self.w1, X1)
        Y2 = torch.matmul(self.w2, X2)
        # print(Y1.size())   # torch.Size([2, 400])

        Y1 = mask_logits(Y1, mask)
        Y2 = mask_logits(Y2, mask)

        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)
        return p1, p2


class MatchLSTM(nn.Module):
    def __init__(self, word_mat, char_mat):
        super().__init__()

        self.embedding_size = 364
        self.hidden_dim = 256
        class_size = 2
        # 加载所谓的词向量, 字符向量
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=Config.pretrained_char)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))

        # 将词向量和字符向量进行揉搓
        self.emb = Embedding()

        self.C_lstm = nn.LSTMCell(self.embedding_size, self.hidden_dim)
        self.Q_lstm = nn.LSTMCell(self.embedding_size, self.hidden_dim)
        self.match_lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.attend_C = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attend_Q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attend_match = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.scale = nn.Linear(self.hidden_dim, 1)

        self.merge_attention = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        self.linear_my = nn.Linear(self.hidden_dim, 96)

        enc_blk = EncoderBlock(conv_num=2, ch_num=d_model, k=5, length=len_c)

        self.model_enc_blks = nn.ModuleList([enc_blk] * 7)  # 七层堆叠的编码块
        self.out = Pointer()

    def initial_hidden_state(self):
        return Variable(torch.zeros([Config.batch_size, self.hidden_dim]))

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        # 对问题和文章进行mask
        cmask = (torch.zeros_like(Cwid) == Cwid).float()
        qmask = (torch.zeros_like(Qwid) == Qwid).float()

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

        C = C.permute((2, 0, 1))
        Q = Q.permute((2, 0, 1))
        # print(C.size())   # (400, 2, 364)

        # 对文章进行lstm  初始化LSTM的初始状态
        hidden_state = self.initial_hidden_state()
        cell_state = self.initial_hidden_state()
        C_states = []
        for c in C:
            hidden_state, cell_state = self.C_lstm(c, (hidden_state, cell_state))
            C_states.append(hidden_state)
        # print(torch.stack(C_states).squeeze(1).size())   # torch.Size([400, 2, 256])

        # 对问题进行lstm
        hidden_state = self.initial_hidden_state()
        cell_state = self.initial_hidden_state()
        Q_states = []  # PlxH
        for q in Q:
            hidden_state, cell_state = self.Q_lstm(q, (hidden_state, cell_state))
            Q_states.append(hidden_state)

        Q_states = torch.stack(Q_states).squeeze(1)  # PlXH
        # print(Q_states.size())   # torch.Size([50, 2, 256])

        # 遍历文章中每步的状态
        My_LIST = []
        hidden_state = self.initial_hidden_state()
        cell_state = self.initial_hidden_state()
        for c in C_states:
            c_attn = self.attend_C(c)   # 1 x hidden_size
            # print(c_attn.size())  # torch.Size([2, 256])
            q_attn = self.attend_Q(Q_states)   # q_len, hidden_size
            # print(q_attn.size())  # torch.Size([50, 2, 256])

            mattn = self.attend_match(hidden_state)

            scale = self.scale(c_attn.expand_as(q_attn) + q_attn + mattn.expand_as(q_attn))
            # print(scale.size())   # torch.Size([50, 2, 1])
            attn = F.softmax(scale).permute(1, 2, 0)
            # print(attn.size())    # torch.Size([2, 1, 50])

            q_attn = q_attn.permute(1, 0, 2)
            # print(q_attn.size())   # torch.Size([2, 50, 256])

            attn = torch.bmm(attn, q_attn)   # 问题向量乘对应的权重值
            # print(attn.size())   # torch.Size([2, 1, 256])     # 问题向量

            c = c.view(Config.batch_size, 1, self.hidden_dim)
            # print(c.size())   # torch.Size([2, 1, 256])
            attn_hidden_mat = self.merge_attention(torch.cat([attn, c], 2))  # HXH
            # print(attn_hidden_mat.size())   # torch.Size([2, 1, 256])

            attn_hidden_mat = attn_hidden_mat.squeeze()
            hidden_state, cell_state = self.match_lstm(attn_hidden_mat, (hidden_state, cell_state))
            # print('hidden_state:{}'.format(hidden_state.size()))
            My_LIST.append(hidden_state)

        My_LIST = torch.stack(My_LIST).squeeze(1)  # PlXH
        # print(My_LIST.size())   # torch.Size([400, 2, 256])

        My_LIST = My_LIST.permute(1, 0, 2)

        My_LIST = self.linear_my(My_LIST)
        # print(My_LIST.size())   # torch.Size([2, 400, 96])


        M1 = My_LIST.permute(0, 2, 1)  # torch.Size([2, 96, 400])

        for enc in self.model_enc_blks:
            M1 = enc(M1, cmask)

        M2 = M1

        for enc in self.model_enc_blks:
            M2 = enc(M2, cmask)
        M3 = M2

        for enc in self.model_enc_blks:
            M3 = enc(M3, cmask)

        # print(M1.size())   # torch.Size([2, 96, 400])
        # print(M2.size())   # torch.Size([2, 96, 400])
        # print(M3.size())   # torch.Size([2, 96, 400])

        p1, p2 = self.out(M1, M2, M3, cmask)
        # print(p1)
        # print(p2)
        # print(p1.size())   # torch.Size([2, 400])
        # print(p2.size())   # torch.Size([2, 400])
        return p1, p2


