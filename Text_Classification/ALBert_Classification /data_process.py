"""

@file  : data_process.py

@author: xiaolu

@time  : 2020-03-19

"""
from transformers import BertTokenizer
import json


def load_data(path):
    '''
    加载数据集
    :param path:
    :return:
    '''
    data_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_list.append(line)
    return data_list


def build_bert_input(data, label):
    tokenizer = BertTokenizer.from_pretrained('./data/vocab.txt')
    content = []
    for d in data:
        d = '[CLS]' + d
        d = tokenizer.tokenize(d)
        d = tokenizer.convert_tokens_to_ids(d)
        seq_len = len(d)
        if len(d) < max_len:
            mask = [1] * len(d) + [0] * (max_len - len(d))
            d += ([0] * (max_len - len(d)))
        else:
            mask = [1] * max_len
            d = d[:max_len]
            seq_len = max_len
        content.append((d, int(label), seq_len, mask))
    return content


if __name__ == '__main__':
    max_len = 10
    # 1. 加载positive
    pos_path = './data/positive.txt'
    pos_label = 0
    pos_data = load_data(pos_path)
    pos_data = build_bert_input(pos_data, pos_label)
    json.dump(pos_data, open('./data/pos_data.json', 'w'))

    # 2. 加载neutral
    neu_path = './data/neutral.txt'
    neu_label = 1
    neu_data = load_data(neu_path)
    neu_data = build_bert_input(neu_data, neu_label)
    json.dump(neu_data, open('./data/neu_data.json', 'w'))

    # 3. 加载negative
    neg_path = './data/negative.txt'
    neg_label = 2
    neg_data = load_data(neg_path)
    neg_data = build_bert_input(neg_data, neg_label)
    json.dump(neg_data, open('./data/neg_data.json', 'w'))

