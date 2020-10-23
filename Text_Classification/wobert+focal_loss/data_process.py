# -*- encoding: utf-8 -*-
'''
@File    :   data_process.py
@Time    :   2020/10/21 17:22:40
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import numpy as np
import jieba
import pandas as pd
from transformers import BertTokenizer
import random
import gzip
import pickle
from tqdm import tqdm
from pdb import set_trace


class CLSExample:
    def __init__(self, doc_id=None, context=None, label=None):
        self.doc_id = doc_id
        self.context = context
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "doc_id: %s" % (str(self.doc_id))
        s += ", context: %s" % (self.context)
        s += ", label: %d" % (self.label)
        return s


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
    
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "input_ids: %s" % (str(self.input_ids))
        s += ", input_mask: %s" % (self.input_mask)
        s += ", segment_ids: %s" % (self.segment_ids)
        s += ", label: %d" % (self.label)
        return s


def convert_example(data):
    labels = data['label'].tolist()
    content = data['review'].tolist()
    doc_id = 0
    examples = []
    for con, lab in zip(content, labels):
        doc_id += 1
        example = CLSExample(doc_id=doc_id, context=con, label=lab)
        examples.append(example)
    return examples


def load_vocab(path):
    vocab = dict()
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            vocab[line] = i
    return vocab


def convert_ids(examples, vocab):
    max_seq_len = 120
    tokenizer = BertTokenizer.from_pretrained('./wobert_pretrain/vocab.txt')

    features = []
    for example in tqdm(examples):
        res = jieba.lcut(example.context)
        # 遍历当前的分词结果 转为id  如果不能转的  可以借助BertTokenizer
        input_ids = []
        input_ids = [3]
        for r in res:
            if vocab.get(r) != None:
                input_ids.append(vocab.get(r))
            else:
                input_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(r)))

        if len(input_ids) > max_seq_len-1:
            input_ids = input_ids[:max_seq_len-1]
            input_ids.append(4)  # add sep
            segment_ids = [0] * len(input_ids)
            attention_mask = [1] * len(input_ids)
            assert len(input_ids) == len(segment_ids) == len(attention_mask)
        else:
            attention_mask = [1] * len(input_ids) + [0] * (max_seq_len - len(input_ids))
            input_ids = input_ids + [0] * (max_seq_len - len(input_ids))
            segment_ids = [0] * max_seq_len
            assert len(input_ids) == len(segment_ids) == len(attention_mask)
        label = int(example.label)
        feature = InputFeatures(input_ids=input_ids, input_mask=attention_mask, segment_ids=segment_ids, label=label)
        features.append(feature)
    return features
        

if __name__ == '__main__':
    # max_length = 120

    data = pd.read_csv('./data/waimai_10k.csv')
    examples = convert_example(data)
    # print(examples[0])

    # 加载词表
    vocab = load_vocab('./wobert_pretrain/vocab.txt')

    # 分词转id
    features = convert_ids(examples, vocab)
    random.shuffle(features)

    dev_features = features[:500]
    train_features = features[500:]

    # 保存
    with gzip.open('./data/train_features.pkl.gz', 'wb') as fout:
        pickle.dump(train_features, fout)

    with gzip.open('./data/dev_features.pkl.gz', 'wb') as fout:
        pickle.dump(dev_features, fout)








    