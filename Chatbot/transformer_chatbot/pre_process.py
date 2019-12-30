"""

@file   : pre_process.py

@author : xiaolu

@time   : 2019-12-26

"""
import pickle
import numpy as np
from tqdm import tqdm
from config import Config
from utils import encode_text


def build_vocab(token):
    '''
    建立词表
    :param token: 某句话的一系列词
    :return:
    '''
    if token not in char2idx:
        next_index = len(char2idx)
        char2idx[token] = next_index
        idx2char[next_index] = token


def process(file):
    '''
    加载语料
    :param file:
    :return:
    '''
    print("processing {} ...".format(file))

    with open(file, 'r', encoding='utf8') as f:
        data = f.readlines()
    # print(data[0])   # 南京在哪里 | 在这里了

    lengths = []
    for line in tqdm(data):
        sentences = line.split('|')
        for sent in sentences:
            sentence = sent.strip()
            lengths.append(len(sentence))
            tokens = list(sentence)
            for token in tokens:
                build_vocab(token)

    np.save('./data/lengths.npy', np.array(lengths))  # 把每条语料的长度保存起来


def get_data(in_file):
    '''
    得到数据并切分数据
    :param in_file:
    :return:
    '''
    print('getting data {}...'.format(in_file))
    with open(in_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    samples = []
    for line in lines:
        sentences = line.split('|')
        in_sentence = sentences[0].strip()
        out_sentence = sentences[1].strip()

        in_data = encode_text(char2idx, in_sentence)
        out_data = [Config.sos_id] + encode_text(char2idx, out_sentence) + [Config.eos_id]

        if len(in_data) < Config.maxlen_in and len(out_data) < Config.maxlen_out and Config.unk_id not in in_data and Config.unk_id not in out_data:
            samples.append({'in': in_data, 'out': out_data})
    return samples


if __name__ == '__main__':
    # 构造词典
    char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}

    process(Config.train_filename)

    # print("词典的大小:", len(char2idx))  # 词典的大小: 5884
    # print("前100个词:", list(char2idx.items())[:100])   # 前100个词: [('<pad>', 0), ('<sos>', 1), ('<eos>', 2)...]

    data = {
        'dict': {
            'char2idx': char2idx,
            'idx2char': idx2char
        }
    }
    # 将词典保存
    with open(Config.vocab_file, 'wb') as file:
        pickle.dump(data, file)

    # 加载样本 并整理
    samples = get_data(Config.train_filename)
    # print(len(samples))   # 总共有116335多条语料

    np.random.shuffle(samples)
    num_samples = len(samples)
    # 从11万条语料中切出1000条当做验证集 10条当做测试集
    num_valid = 1000
    num_test = 10

    valid = samples[:num_valid]
    test = samples[num_valid:num_valid + num_test]
    train = samples[num_valid + num_test:]

    data = {
        'train': train,
        'valid': valid,
        'test': test
    }
    print("训练集条数:", len(train))
    print("验证集条数:", len(valid))
    print("测试集条数:", len(test))

    # 将数据保存
    with open(Config.data_file, 'wb') as file:
        pickle.dump(data, file)

