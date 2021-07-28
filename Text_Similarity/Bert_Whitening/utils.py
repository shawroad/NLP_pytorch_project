"""
@file   : utils.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-28$
"""
import pickle
import numpy as np
from config import set_args


args = set_args()


def load_STS_data(path):
    data = []
    with open(path) as f:
        for i in f:
            d = i.split("||")
            sentence1 = d[1]
            sentence2 = d[2]
            score = int(d[3])
            data.append([sentence1, sentence2, score])
    return data


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s ** 0.5))
    W = np.linalg.inv(W.T)
    return W, -mu


def save_whiten(path, kernel, bias):
    whiten = {
        'kernel': kernel,
        'bias': bias
    }
    with open(path, 'wb') as f:
        pickle.dump(whiten, f)
    return path


def load_whiten(path):
    with open(path, 'rb') as f:
        whiten = pickle.load(f)
    kernel = whiten['kernel']
    bias = whiten['bias']
    return kernel, bias


def normalize(vecs):
    """
    标准化
    """
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


def transform_and_normalize(vecs, kernel, bias):
    """
    应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel[:, :args.dim])
    return normalize(vecs)
