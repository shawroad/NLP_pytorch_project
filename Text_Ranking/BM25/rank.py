# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 10:33
# @Author  : xiaolu
# @FileName: rank.py
# @Software: PyCharm

from math import log

k1 = 1.2
k2 = 100
b = 0.75
R = 0.0


def score_BM25(n, f, qf, r, N, dl, avdl):
    '''
    :param n: 当前词在多少文档中出现过
    :param f: 当前词在当前文档中出现的次数
    :param qf: 1
    :param r: 0
    :param N: 多少篇文章
    :param dl: 当前这篇文章的长度
    :param avdl: 所有文章的平均长度
    :return:
    '''
    K = compute_K(dl, avdl)   # 文章越长 这个K值越大

    first = log(((r + 0.5) / (R - r + 0.5)) / ((n - r + 0.5) / (N - n - R + r + 0.5)))
    second = ((k1 + 1) * f) / (K + f)
    third = ((k2 + 1) * qf) / (k2 + qf)   # 按目前这种方式 这个值对于任何语料都是相等
    return first * second * third


def compute_K(dl, avdl):
    # (1-0.75) * 1.2 + 0.75 * (当前这篇文章的长度 / 所有文章的平均长度)
    return k1 * ((1 - b) + b * (float(dl) / float(avdl)))
