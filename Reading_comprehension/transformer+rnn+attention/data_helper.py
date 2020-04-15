"""

@file  : data_helper.py

@author: xiaolu

@time  : 2020-04-15

"""
import pandas as pd
import json
import numpy as np
from config import Config


def build_vocab(data):
    '''
    建立词表
    :param data:
    :return:
    '''
    context = data['context'].tolist()
    question = data['question'].tolist()
    answer = data['answer'].tolist()

    total_str = ''
    for c, q, a in zip(context, question, answer):
        total_str += c
        total_str += q
        total_str += a

    vocab = list(set(list(total_str)))
    print(len(vocab))   # 2668

    PAD = 0  # Used for padding short sentences
    SOS = 1  # Start-of-sentence token
    EOS = 2  # End-of-sentence token
    UNK = 3

    vocab2id = dict()
    vocab2id['PAD'] = PAD
    vocab2id['SOS'] = SOS
    vocab2id['EOS'] = EOS
    vocab2id['UNK'] = UNK
    id2vocab = dict()
    id2vocab[PAD] = 'PAD'
    id2vocab[SOS] = 'SOS'
    id2vocab[EOS] = 'EOS'
    id2vocab[UNK] = 'UNK'

    for i, v in enumerate(vocab):
        vocab2id[v] = i + 4
        id2vocab[i+4] = v
    vocab_size = len(vocab2id)

    return vocab2id, id2vocab, vocab_size


def convert_id(data, vocab2id):
    '''
    :param data:
    :param vocab2id:
    :return:
    '''
    context = data['context'].tolist()
    question = data['question'].tolist()
    answer = data['answer'].tolist()

    # 处理输入数据
    max_len = 0
    for c, q in zip(context, question):
        s = c + q
        if len(s) > max_len:
            max_len = len(s)
    # print(max_len)     # 492

    input_data = []
    input_len = []
    for c, q in zip(context, question):
        s = c + q
        ids = [vocab2id.get(i, 'UNK') for i in s]

        if len(ids) < max_len:
            ids = ids + [vocab2id.get("PAD")] * (max_len - len(ids))
        input_len.append(len(s))
        input_data.append(ids)

    # 处理输出数据
    max_ans_len = 0
    for a in answer:
        if len(a) > max_ans_len:
            max_ans_len = len(a)
    # print(max_ans_len)   # 答案的最大长度62
    max_ans_len += 1  # 因为还要加结束标志

    output_data = []
    mask = []
    output_len = []
    for a in answer:
        ids = [vocab2id.get(i, 'UNK') for i in a]
        ids = ids + [vocab2id.get('EOS')]
        output_len.append(len(ids))
        m = len(ids) * [1]
        m = m + (max_ans_len - len(ids)) * [0]


        if len(ids) < max_ans_len:
            ids = ids + [vocab2id.get('PAD')] * (max_ans_len - len(ids))
        output_data.append(ids)
        mask.append(m)

    # 按照原始输入数据的长短进行排序 从长到短
    length = np.array(input_len)
    ids = length.argsort()
    ids = ids.tolist()
    ids = list(reversed(ids))

    f_input = []
    f_input_l = []
    f_output = []
    f_mask = []
    f_output_l = []
    for i in ids:
        f_input.append(input_data[i])
        f_input_l.append(input_len[i])
        f_output.append(output_data[i])
        f_mask.append(mask[i])
        f_output_l.append(output_len[i])

    data = {
        'input_data': f_input,
        'input_len': f_input_l,
        'output_data': f_output,
        'mask': f_mask,
        'output_len': f_output_l
    }

    with open('./data/train.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':

    path = './data/correct_data.csv'
    data = pd.read_csv(path)

    vocab2id, id2vocab, vocab_size = build_vocab(data)

    # 将输入输出数据转为id并进行padding
    convert_id(data, vocab2id)



