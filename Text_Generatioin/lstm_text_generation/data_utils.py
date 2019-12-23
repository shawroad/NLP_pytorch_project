"""

@file  : data_utils.py

@author: xiaolu

@time  : 2019-11-06

"""
import torch


class Dictionary:
    # 整理字典
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx  # 加入新词的时候　我们接着给其标号
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus:
    '''
    加载语料　并整理语料
    '''
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        # 加载语料
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)   # 给当前这个词编号 也就是整理一个词典

        # print(tokens)  # 929589
        ids = torch.LongTensor(tokens)   # 应该就是总的词个数(未取重)
        token = 0

        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]  # 将对应的句子转为id序列
                    token += 1
        # print(ids.size())  # torch.Size([929589])

        num_batches = ids.size(0) // batch_size  # 句子总的条数　// batch_size
        ids = ids[:num_batches * batch_size]
        return ids.view(batch_size, -1)   # [batch_size, 句子长度]
