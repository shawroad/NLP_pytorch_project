"""
@file   : data_process.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-04-13
"""
import numpy as np
from tqdm import tqdm


class BOW(object):
    def __init__(self, X, min_count=10, maxlen=100):
        """
        X: [[w1, w2],]]
        """
        self.X = X
        self.min_count = min_count
        self.maxlen = maxlen
        self.__word_count()
        self.__idx()
        self.__doc2num()

    def __word_count(self):
        wc = {}
        for ws in tqdm(self.X, desc='   Word Count'):
            for w in ws:
                if w in wc:
                    wc[w] += 1
                else:
                    wc[w] = 1
        self.word_count = {i: j for i, j in wc.items() if j >= self.min_count}

    def __idx(self):
        self.idx2word = {i + 1: j for i, j in enumerate(self.word_count)}
        self.word2idx = {j: i for i, j in self.idx2word.items()}

    def __doc2num(self):
        doc2num = []
        for text in tqdm(self.X, desc='Doc To Number'):
            s = [self.word2idx.get(i, 0) for i in text[:self.maxlen]]
            doc2num.append(s + [0]*(self.maxlen-len(s)))  # 未登录词全部用0表示
        self.doc2num = np.asarray(doc2num)
