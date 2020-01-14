"""

@file  : QANet.py

@author: xiaolu

@time  : 2020-01-11

"""
"""

@file  : QANet.py

@author: xiaolu

@time  : 2020-01-09

"""
from torch import nn
import torch
import torch.nn.functional as F
import math
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


def mask_logits(target, mask):
    return target * (1-mask) + mask * (-1e30)


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


class CQAttention(nn.Module):
    '''
    问题和文章注意力交互
    '''
    def __init__(self):
        super().__init__()
        w = torch.empty(Config.d_model * 3)
        lim = 1 / Config.d_model
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)

    def forward(self, C, Q, cmask, qmask):
        ss = []
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        # print(C.size())   # torch.Size([2, 400, 96])
        # print(Q.size())   # torch.Size([2, 50, 96])

        cmask = cmask.unsqueeze(2)
        qmask = qmask.unsqueeze(1)
        # print(cmask.size())  # torch.Size([2, 400, 1])
        # print(qmask.size())  # torch.Size([2, 1, 50])

        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))

        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        # print(Ct.size())   # torch.Size([2, 400, 50, 96])
        # print(Qt.size())   # torch.Size([2, 400, 50, 96])

        CQ = torch.mul(Ct, Qt)   # 对应维度的数据成对应维度(不是矩阵乘法)
        # print(CQ.size())   # torch.Size([2, 400, 50, 96])

        S = torch.cat([Ct, Qt, CQ], dim=3)
        # print(S.size())     # torch.Size([2, 400, 50, 288])

        S = torch.matmul(S, self.w)
        # print(S.size())   # torch.Size([2, 400, 50])

        S1 = F.softmax(mask_logits(S, qmask), dim=2)
        S2 = F.softmax(mask_logits(S, cmask), dim=1)
        # print(S1.size())   # torch.Size([2, 400, 50])
        # print(S2.size())   # torch.Size([2, 400, 50])

        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        out = F.dropout(out, p=Config.dropout, training=self.training)
        return out.transpose(1, 2)


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
    '''
    def __init__(self):
        super().__init__()
        self.conv2d = DepthwiseSeparableConv(d_char, d_char, 5, dim=2)
        self.high = Highway(2, d_word+d_char)

    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
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


class QANet(nn.Module):

    def __init__(self, word_mat, char_mat):
        super().__init__()
        # 加载所谓的词向量, 字符向量
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=Config.pretrained_char)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))

        # 将词向量和字符向量进行揉搓
        self.emb = Embedding()

        self.context_conv = DepthwiseSeparableConv(d_word + d_char, d_model, 5)
        self.question_conv = DepthwiseSeparableConv(d_word + d_char, d_model, 5)

        self.c_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_c)
        self.q_emb_enc = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_q)

        self.cq_att = CQAttention()   # 文章和问题注意力交互

        self.cq_resizer = DepthwiseSeparableConv(d_model * 4, d_model, 5)

        enc_blk = EncoderBlock(conv_num=2, ch_num=d_model, k=5, length=len_c)

        self.model_enc_blks = nn.ModuleList([enc_blk] * 7)   # 七层堆叠的编码块
        self.out = Pointer()

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
        # print(Cw)
        # print(Cc)

        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        # print(Qw.size())   # torch.Size([2, 50, 300])
        # print(Qc.size())   # torch.Size([2, 50, 16, 64])

        # 将文章的词向量,字符向量进行融合　　　将问题的词向量,字符向量进行融合
        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        # print(C.size())   # torch.Size([2, 364, 400])
        # print(Q.size())   # torch.Size([2, 364, 50])

        # 将文章和问题一顿卷
        C = self.context_conv(C)
        Q = self.question_conv(Q)
        # print(C.size())   # torch.Size([2, 96, 400])
        # print(Q.size())   # torch.Size([2, 96, 50])

        Ce = self.c_emb_enc(C, cmask)
        Qe = self.q_emb_enc(Q, qmask)
        # print(Ce.size())   # torch.Size([2, 96, 400])
        # print(Qe.size())   # torch.Size([2, 96, 50])

        # 文章和问题注意力交互
        X = self.cq_att(Ce, Qe, cmask, qmask)
        # print(X.size())    # torch.Size([2, 384, 400])

        M1 = self.cq_resizer(X)  # 再来一波卷积
        # print(M1.size())   # torch.Size([2, 96, 400])

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