"""
# -*- coding: utf-8 -*-
# @File    : inference.py
# @Time    : 2020/12/9 1:45 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
import json
from pdb import set_trace
import numpy as np
from config import set_args
from model import s_model, po_model


if __name__ == '__main__':
    args = set_args()
    # 加载类别  词表
    id2predicate, predicate2id = json.load(open('./data/all_50_schemas_me.json'))
    id2predicate = {int(i): j for i, j in id2predicate.items()}
    id2char, char2id = json.load(open('./data/all_chars_me.json'))
    num_classes = len(id2predicate)
    s_model = s_model(len(char2id) + 2, args.char_size, args.hidden_size)
    po_model = po_model(len(char2id) + 2, args.char_size, args.hidden_size, 49)

    text_in = '周杰伦出演了电影不能说的秘密。我当时是在电影院看的'
    t_s = [char2id.get(i, 1) for i in text_in]
    _s = torch.LongTensor([t_s])

    R = []
    with torch.no_grad():
        _k1, _k2, t, t_max, mask = s_model(torch.LongTensor(_s))
        _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
        _kk1s = []
        for i, _kk1 in enumerate(_k1):
            if _kk1 > 0.5:
                _subject = ''
                for j, _kk2 in enumerate(_k2[i:]):
                    if _kk2 > 0.5:
                        _subject = text_in[i: i + j + 1]
                        break
                if _subject:
                    _k1, _k2 = torch.LongTensor([[i]]), torch.LongTensor([[i + j]])  # np.array([i]), np.array([i+j])
                    _o1, _o2 = po_model(t.cuda(), t_max.cuda(), _k1.cuda(), _k2.cuda())
                    _o1, _o2 = _o1.cpu().data.numpy(), _o2.cpu().data.numpy()
                    _o1, _o2 = np.argmax(_o1[0], 1), np.argmax(_o2[0], 1)

                    for i, _oo1 in enumerate(_o1):
                        if _oo1 > 0:
                            for j, _oo2 in enumerate(_o2[i:]):
                                if _oo2 == _oo1:
                                    _object = text_in[i: i + j + 1]
                                    _predicate = id2predicate[_oo1]
                                    # print((_subject, _predicate, _object))
                                    R.append((_subject, _predicate, _object))
                                    break
            _kk1s.append(_kk1.data.cpu().numpy())
        _kk1s = np.array(_kk1s)
        print(list(set(R)))






