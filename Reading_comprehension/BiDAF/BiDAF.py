"""

@file   : BiDAF.py

@author : xiaolu

@time   : 2020-02-16

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Linear(nn.Module):
    '''
    带有dropout  参数初始化 的 全连接
    '''
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

        # 参数初始化
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


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


def mask_logits(target, mask):
    return target * (1-mask) + mask * (-1e30)


class BiDAF(nn.Module):
    def __init__(self, word_mat, char_mat):
        super().__init__()
        # 加载所谓的词向量, 字符向量
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=Config.pretrained_char)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))

        # 将词向量和字符向量进行揉搓
        self.emb = Embedding()

        # 3. Contextual Embedding Layer
        # self.context_LSTM = LSTM(input_size=364, hidden_size=200, bidirectional=True, batch_first=True, dropout=0.3)
        # self.lstm = LSTM(input_size=364, hidden_size=512, bidirectional=False, batch_first=True)
        self.lstm = nn.LSTM(input_size=364, hidden_size=512, bidirectional=False, batch_first=True)

        self.att_weight_c = nn.Linear(512, 1)
        self.att_weight_q = nn.Linear(512, 1)
        self.att_weight_cq = nn.Linear(512, 1)

        self.model_lstm1 = nn.LSTM(input_size=2048, hidden_size=256, batch_first=True, dropout=0.3, bidirectional=True)
        self.model_lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True, dropout=0.2, bidirectional=True)

        self.p1_weight_g = Linear(256 * 8, 1, dropout=0.3)
        self.p1_weight_m = Linear(256 * 2, 1, dropout=0.3)
        self.p2_weight_g = Linear(256 * 8, 1, dropout=0.3)
        self.p2_weight_m = Linear(256 * 2, 1, dropout=0.3)

        self.output_LSTM = nn.LSTM(input_size=2 * 256, hidden_size=256,
                                   bidirectional=True, batch_first=True, dropout=0.3)

        self.dropout = nn.Dropout(p=0.3)

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

        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        # print(Qw.size())   # torch.Size([2, 50, 300])
        # print(Qc.size())   # torch.Size([2, 50, 16, 64])

        # 将文章的词向量,字符向量进行融合　　　将问题的词向量,字符向量进行融合
        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        # print(C.size())   # torch.Size([2, 364, 400])
        # print(Q.size())   # torch.Size([2, 364, 50])
        # exit()

        C = torch.transpose(C, 2, 1)
        Q = torch.transpose(Q, 2, 1)
        # print(C.size())    # torch.Size([2, 400, 364]) (batch_size, max_Len, embedding_size)
        # print(Q.size())    # torch.Size([2, 50, 364]) (batch_size, max_Len, embedding_size)

        C, _ = self.lstm(C)
        Q, _ = self.lstm(Q)
        # print(C.size())   # torch.Size([2, 400, 512])
        # print(Q.size())   # torch.Size([2, 50, 512])

        def attn_flow_layer(c, q):
            '''
            :param c: (batch, c_len, 512)
            :param q: (batch, q_len, 512)
            :return: (batch, c_len, q_len
            '''
            c_len = c.size(1)
            q_len = q.size(1)

            cq = []
            for i in range(q_len):
                qi = q.select(1, i).unsqueeze(1)   # batch, 1, hidden_size *2

                # 上下文和问题中的每个词求相似度
                ci = self.att_weight_cq(c * qi).squeeze()   # torch.Size([2, 400]) batch_size, context_max_len
                cq.append(ci)

            cq = torch.stack(cq, dim=-1)   # torch.Size([2, 400, 50])   # 文本中每个词 对问题中每个词都有一个概率值
            # print(cq.size())   # (batch, c_len, q_len)

            s = self.att_weight_c(c).expand(-1, -1, q_len) + self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + cq
            # print(s.size())   # (batch, c_len, q_len)

            a = F.softmax(s, dim=2)

            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # print(c2q_att.size())   # torch.Size([2, 400, 512])

            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # print(b.size())   # torch.Size([2, 1, 400])

            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()   # torch.Size([2, 512])
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)

            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            # print(x.size())   # torch.Size([2, 400, 2048])
            return x

        g = attn_flow_layer(C, Q)   # torch.Size([2, 400, 2048])

        m = self.model_lstm2(self.model_lstm1(g)[0])[0]
        # print(m.size())   # torch.Size([2, 400, 512])

        def output_layer(g, m):
            '''
            :param g: (batch, c_len, hidden_size * 8)   hidden_size: 256
            :param m: (batch, c_len ,hidden_size * 2)
            :return:  p1: (batch, c_len), p2: (batch, c_len)
            '''
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            # print(p1.size())   # torch.Size([2, 400])

            m2 = self.output_LSTM(m)[0]
            # print(m2.size())   # torch.Size([2, 400, 512])

            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()

            return p1, p2

        p1, p2 = output_layer(g, m)

        Y1 = mask_logits(p1, cmask)
        Y2 = mask_logits(p2, cmask)

        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)
        print(p1)
        print(p2)
        exit()


        return p1, p2

