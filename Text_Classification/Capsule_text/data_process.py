"""

@file  : data_process.py

@author: xiaolu

@time  : 2020-06-04

"""
import json


def load_data(path):
    texts, labels = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label, text = line.split("	")
            texts.append(text)
            labels.append(int(label))
    return texts, labels


if __name__ == '__main__':
    path = './data/train.tsv'
    texts, labels = load_data(path)

    # 1. 建立字典
    total_text = ''.join(texts)
    vocab = list(set(total_text))
    # print(len(vocab))  # 4501

    vocab2id = {'PAD': 0}
    vocab2id['UNK'] = 1
    for i, v in enumerate(vocab):
        vocab2id[v] = i+1
    vocab_size = len(vocab2id)
    # print(len(vocab2id))  # 4502

    max_len = max([len(text) for text in texts])
    # print(max_len)   # 225

    # 2. 将文本转为id　并进行padding
    sentence_ids = []
    sentence_len = []
    for text in texts:
        ids = [vocab2id[i] for i in text]
        sentence_len.append(len(ids))
        ids += (max_len - len(ids)) * [0]
        sentence_ids.append(ids)
    # print(len(sentence_ids))  # 10000
    # print(len(sentence_len))  # 10000
    # print(len(labels))   # 10000

    data = {'sentence_ids': sentence_ids,
            'sentence_len': sentence_len,
            'labels': labels}

    json.dump(data, open('./data/train.json', 'w'))
    json.dump(vocab2id, open('./data/vocab2id.json', 'w'))









