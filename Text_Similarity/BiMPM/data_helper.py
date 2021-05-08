"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-07
"""
import re
import gensim
import torch
import numpy as np
from torch.utils.data import Dataset


class LCQMC_Dataset(Dataset):
    def __init__(self, data_file, vocab_file, max_char_len):
        # 1. 加载数据
        p, h, self.label = self.load_sentences(data_file)

        # 2. 加载词表
        word2idx, _, _ = self.load_vocab(vocab_file)

        # 3. 将数据转为id序列
        self.p_list, self.p_lengths, self.h_list, self.h_lengths = self.word_index(p, h, word2idx, max_char_len)
        self.p_list = torch.tensor(self.p_list, dtype=torch.long)
        self.h_list = torch.tensor(self.h_list, dtype=torch.long)
        self.max_len = max_char_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.p_list[idx], self.p_lengths[idx], self.h_list[idx], self.h_lengths[idx], self.label[idx]

    def load_sentences(self, file):
        '''
        加载数据 并进行分词
        :param file:
        :return:
        '''
        p_list, h_list, label_list = [], [], []
        with open(file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                p, h, lab = line.strip().split('\t')
                p_list.append(p)
                h_list.append(h)
                label_list.append(int(lab))
        p_list = map(get_word_list, p_list)
        h_list = map(get_word_list, h_list)
        return p_list, h_list, label_list

    def load_vocab(self, vocab_file):
        '''
        加载词表
        :param vocab_file:
        :return:
        '''
        vocab = [line.strip() for line in open(vocab_file, encoding='utf-8').readlines()]
        word2idx = {word: index for index, word in enumerate(vocab)}
        idx2word = {index: word for index, word in enumerate(vocab)}
        return word2idx, idx2word, vocab

    def word_index(self, p_sentences, h_sentences, word2idx, max_char_len):
        p_list, p_length, h_list, h_length = [], [], [], []
        for p_sentence, h_sentence in zip(p_sentences, h_sentences):
            p = [word2idx[word] for word in p_sentence if word in word2idx.keys()]
            h = [word2idx[word] for word in h_sentence if word in word2idx.keys()]
            p_list.append(p)
            p_length.append(min(len(p), max_char_len))
            h_list.append(h)
            h_length.append(min(len(h), max_char_len))
        p_list = pad_sequences(p_list, maxlen=max_char_len)
        h_list = pad_sequences(h_list, maxlen=max_char_len)
        return p_list, p_length, h_list, h_length


def get_word_list(query):
    regEx = re.compile('[\\W]+')   # 正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')   #[\u4e00-\u9fa5]中文范围
    sentences = regEx.split(query.lower())
    str_list = []
    for sentence in sentences:
        if res.split(sentence) is None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。
    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def load_embeddings(embdding_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(embdding_path, binary=False)
    embedding_matrix = np.zeros((len(model.index2word) + 1, model.vector_size))
    # 填充向量矩阵
    for idx, word in enumerate(model.index2word):
        embedding_matrix[idx + 1] = model[word]   # 词向量矩阵
    return embedding_matrix