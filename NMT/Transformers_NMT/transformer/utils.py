"""

@file  : utils.py

@author: xiaolu

@time  : 2019-12-25

"""
import torch


def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """
    padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])
        for i in range(N):
            # 相当于句子长度不一 我们调整成同样的长度　后面padding全部为0 加这一步就是为了防止有些人padding不是用零进行填充的
            non_pad_mask[i, input_lengths[i]:] = 0

    if pad_idx is not None:
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()  # 把padding的标志全部置为false

    return non_pad_mask.unsqueeze(-1)


def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """
    mask position is set to 1
    """
    # batch_size x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)

    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask


def pad_list(xs, pad_value):
    '''
    :param xs: 一个id序列
    :param pad_value: 要填充的值
    :return:
    '''
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


def get_subsequent_mask(seq):
    '''
    :param seq: 解码的输入序列 batch_size x max_len
    :return:
    example:  假设有一个序列[1, 3, 4] 　此时真实length=3 假设我们的其实标志为0  结束标志为9  maxlen=7
    则对应这个函数的输入[0, 1, 3, 4, 9, 9, 9]
    '''
    sz_b, len_s = seq.size()
    # diagonal标志的上三角是否向右移动　值为1 则代表向右偏移一位
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)

    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    '''
    For masking out the padding part of key sequence.
    '''
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)  # 等于pad_idx置为True　否则置为False
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


if __name__ == '__main__':
    data = torch.randint(0, 20, (5, 10))
    result = get_attn_key_pad_mask(data, data, pad_idx=2)
    print(result)


