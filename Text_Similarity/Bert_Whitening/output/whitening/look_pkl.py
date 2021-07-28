"""
@file   : look_pkl.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-28$
"""
import pickle
from pdb import set_trace


def load_whiten(path):
    with open(path, 'rb') as f:
        whiten = pickle.load(f)
    kernel = whiten['kernel']
    bias = whiten['bias']
    return kernel, bias


if __name__ == '__main__':
    path = 'first_last-whiten.pkl'
    k, b = load_whiten(path)
    # k [[], [], []]  长度为768 里面的每个元素也是长度为768的列表
    # b [[]]   长度为1(列表中一个元素), 里面那个元素是长度为768的列表
    set_trace()


