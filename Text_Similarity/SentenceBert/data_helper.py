"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-02
"""
import torch
import gzip
import pickle
import pandas as pd
import random
from tqdm import tqdm
from transformers import AutoTokenizer


class Features:
    def __init__(self, s1_input_ids=None, s2_input_ids=None, label=None):
        self.s1_input_ids = s1_input_ids
        self.s2_input_ids = s2_input_ids
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "s1_input_ids: %s" % (self.s1_input_ids)
        s += ", s2_input_ids: %s" % (self.s2_input_ids)
        s += ", label: %d" % (self.label)
        return s


def get_max_len():
    max_len = 0
    for s, t in tqdm(zip(train_data['source'].tolist(), train_data['target'].tolist())):
        if len(s) > max_len:
            max_len = len(s)
        if len(t) > max_len:
            max_len = len(t)
    return max_len


def convert_token_to_id(data):
    '''
    将句子转为id序列
    :return:
    '''
    features = []
    for s, t, lab in tqdm(zip(data['source'].tolist(), data['target'].tolist(), data['label'].tolist())):
        s_input = tokenizer.encode(s)
        t_input = tokenizer.encode(t)

        if len(s_input) > max_len:
            s_input = s_input[:max_len]
        else:
            s_input = s_input + (max_len - len(s_input)) * [0]

        if len(t_input) > max_len:
            t_input = t_input[:max_len]
        else:
            t_input = t_input + (max_len - len(t_input)) * [0]
        lab = int(lab)
        feature = Features(s1_input_ids=s_input, s2_input_ids=t_input, label=lab)
        features.append(feature)
    return features


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('./pretrain_weight')


    max_len = 128
    data = pd.read_csv('./data/corpus.csv')
    all_features = convert_token_to_id(data)

    random.seed(43)
    random.shuffle(all_features)
    
    with gzip.open('./data/train_features.pkl.gz', 'wb') as fout:
        pickle.dump(all_features, fout)
