"""
# -*- coding: utf-8 -*-
# @File    : data_helper.py
# @Time    : 2020/12/8 1:43 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import gzip
import pickle


class Example(object):
    def __init__(self, text, position1, position2, label):
        self.text = text
        self.position1 = position1
        self.position2 = position2
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "text: %s" % (self.text)
        s += ", position1: %s" % (self.position1)
        s += ", position2: %s" % (self.position2)
        s += ", label: %s" % (self.label)
        return s


class Features(object):
    def __init__(self, input_ids, position1, position2, label_id):
        self.input_ids = input_ids
        self.position1 = position1
        self.position2 = position2
        self.label_id = label_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "input_ids: %s" % (self.input_ids)
        s += ", position1: %s" % (self.position1)
        s += ", position2: %s" % (self.position2)
        s += ", label_id: %s" % (self.label_id)
        return s


def load_data(path, n):
    # 加载数据  并构建example
    with open(path, 'r', encoding='utf8') as f:
        examples = []
        print_log = 0
        # 这里分别搞100
        for line in f.readlines()[:n]:
            print_log += 1
            sub, obj, rel, text = line.split()
            index1 = text.find(sub)    # 第一个实体在文章中的位置
            index2 = text.find(obj)    # 第二个实体在文章中的位置
            sentence, position1, position2 = [], [], []
            for i, word in enumerate(text):
                sentence.append(word)
                position1.append(i - index1)
                position2.append(i - index2)
            example = Example(text=sentence, position1=position1, position2=position2, label=rel)

            if print_log < 10:
                print(example)
            examples.append(example)
    return examples


def load_label(path):
    # label和id进行映射
    relation2id = {}
    id2relation = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            label, id = line.split()
            id = int(id)
            relation2id[label] = id
            id2relation[id] = label
    return relation2id, id2relation


def build_vocab(train_examples, test_examples):
    # 建立词表
    vocab = []
    for train, test in zip(train_examples, test_examples):
        vocab.extend(train.text)
        vocab.extend(test.text)
    # 统计词频  然后建立词表
    vocab_count = {}
    for v in vocab:
        if vocab_count.get(v) is None:
            vocab_count[v] = 1
        else:
            vocab_count[v] = vocab_count[v] + 1
    vocab_count = sorted(vocab_count.items(), key=lambda k: k[1], reverse=True)
    vocab2id = {'pad': 0, 'unk': 1}
    id2vocab = {0: 'pad', 1: 'unk'}
    # 根据词频建立词表
    i = 2
    for v, c in vocab_count:
        vocab2id[v] = i
        id2vocab[i] = v
        i += 1
    return vocab2id, id2vocab


def pos(num):
    if num < -50:
        return 0
    if num >= -50 and num <= 50:
        return num+50
    if num > 50:
        return 100


def position_padding(words, max_len):
    words = [pos(i) for i in words]
    if len(words) >= max_len:
        return words[:max_len]
    words.extend([101]*(max_len-len(words)))
    return words


def convert_features_id(examples, relation2id, vocab2id):
    max_len = 80
    features = []
    print_log = 0
    for e in examples:
        print_log += 1
        text, position1, position2, label = e.text, e.position1, e.position2, e.label
        # 1. 想将text转为id序列
        input_ids = []
        for t in text:
            id_ = vocab2id.get(t)
            if id_ is None:
                id_ = vocab2id.get('unk')
            input_ids.append(id_)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            position1 = position_padding(position1, max_len)
            position2 = position_padding(position2, max_len)
        else:
            input_ids = input_ids + [0] * (max_len - len(input_ids))
            position1 = position_padding(position1, max_len)
            position2 = position_padding(position2, max_len)
        assert len(input_ids) == len(position1) and len(input_ids) == len(position2)
        label_id = relation2id[label]
        feature = Features(input_ids=input_ids, position1=position1, position2=position2, label_id=label_id)
        if print_log < 10:
            print(feature)
        features.append(feature)
    return features


if __name__ == '__main__':
    train_path = './data/train_data.txt'
    test_path = './data/test_data.txt'
    # 将构建example
    print('开始构建examples...')
    train_nums = 10000
    train_examples = load_data(train_path, train_nums)
    with gzip.open('./data/train_examples.pkl.gz', 'wb') as fout:
        pickle.dump(train_examples, fout)
    test_nums = 1000
    test_examples = load_data(test_path, test_nums)
    with gzip.open('./data/test_examples.pkl.gz', 'wb') as fout:
        pickle.dump(test_examples, fout)

    print('开始构建词表, 标签映射')
    # 接下来转features 这里得先建立词表
    relation2id, id2relation = load_label('./data/relation2id.txt')
    vocab2id, id2vocab = build_vocab(train_examples, test_examples)
    print('此表大小(重新构建词表  记得去config中改vocab_size): ', len(vocab2id))

    print('开始构建features...')
    train_features = convert_features_id(train_examples, relation2id, vocab2id)
    with gzip.open('./data/train_features.pkl.gz', 'wb') as fout:
        pickle.dump(train_features, fout)
    test_features = convert_features_id(test_examples, relation2id, vocab2id)
    with gzip.open('./data/test_features.pkl.gz', 'wb') as fout:
        pickle.dump(test_features, fout)










