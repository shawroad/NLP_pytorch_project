"""

@file  : process_data.py

@author: xiaolu

@time  : 2019-12-26

"""
'''
主要功能:
   1. 对中文文本和英文文本建立词典 然后存到vocab.pkl
   2. 将训练集和验证集转化为id序列 然后存到data.pkl
'''
import os
import pickle
from collections import Counter
import jieba
import nltk
from tqdm import tqdm
from config import Config
from utils import normalizeString, encode_text


def build_vocab(token, word2idx, idx2char):
    if token not in word2idx:
        next_index = len(word2idx)
        word2idx[token] = next_index
        idx2char[next_index] = token


def process(file, lang='zh'):
    '''
    建立词表
    :param file:
    :param lang:
    :return:
    '''
    print('processing {}...'.format(file))
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    word_freq = Counter()
    lengths = []

    for line in tqdm(data):
        sentence = line.strip()
        if lang == 'en':
            # 若是英文 转小写 然后切分
            sentence_en = sentence.lower()
            tokens = [normalizeString(s) for s in nltk.word_tokenize(sentence_en)]  # 得到token然后再清洗
            word_freq.update(list(tokens))
            vocab_size = Config.n_src_vocab    # 是由超参数给出的
        else:
            # 若是中文 使用jieba进行分词
            seg_list = jieba.cut(sentence.strip())
            tokens = list(seg_list)
            word_freq.update(list(tokens))
            vocab_size = Config.n_tgt_vocab

        lengths.append(len(tokens))  # 得到每个句子的真实长度

    words = word_freq.most_common(vocab_size - 4)  # vocab_size 统计出词频最高的这么多个词
    word_map = {k[0]: v + 4 for v, k in enumerate(words)}   # 词->id
    word_map['<pad>'] = 0
    word_map['<sos>'] = 1
    word_map['<eos>'] = 2
    word_map['<unk>'] = 3
    print(len(word_map))
    print(words[:100])

    word2idx = word_map
    idx2char = {v: k for k, v in word2idx.items()}

    return word2idx, idx2char


def get_data(in_file, out_file):
    '''
    加载语料
    :param in_file: 中文数据集路径
    :param out_file: 英文数据集路径
    :return:
    '''
    print('getting data {}->{}...'.format(in_file, out_file))
    with open(in_file, 'r', encoding='utf-8') as file:
        in_lines = file.readlines()
    with open(out_file, 'r', encoding='utf-8') as file:
        out_lines = file.readlines()

    samples = []

    for i in tqdm(range(len(in_lines))):
        sentence_zh = in_lines[i].strip()
        tokens = jieba.cut(sentence_zh.strip())
        in_data = encode_text(src_char2idx, tokens)  # encode_text(src_char2idx, tokens) 将语料转为id序列

        sentence_en = out_lines[i].strip().lower()
        tokens = [normalizeString(s.strip()) for s in nltk.word_tokenize(sentence_en)]  # 将英文单词预处理
        out_data = [Config.sos_id] + encode_text(tgt_char2idx, tokens) + [Config.eos_id]   # 转为id　并加上开始和结束标志

        # 这里的maxlen_in=50 和 maxlen_out=100 也是有超参数给出的
        if len(in_data) < Config.maxlen_in and len(out_data) < Config.maxlen_out and Config.unk_id not in in_data and Config.unk_id not in out_data:
            samples.append({'in': in_data, 'out': out_data})
    return samples


if __name__ == '__main__':
    # 加载词表　没有的话　我们建立词表
    if os.path.isfile(Config.vocab_file):
        with open(Config.vocab_file, 'rb') as file:
            data = pickle.load(file)

        src_char2idx = data['dict']['src_char2idx']
        src_idx2char = data['dict']['src_idx2char']
        tgt_char2idx = data['dict']['tgt_char2idx']
        tgt_idx2char = data['dict']['tgt_idx2char']

    else:
        src_char2idx, src_idx2char = process(Config.train_translation_zh_filename, lang='zh')
        tgt_char2idx, tgt_idx2char = process(Config.train_translation_en_filename, lang='en')

        print("输入文本字典的大小:", len(src_char2idx))
        print("输出文本字典的大小:", len(tgt_char2idx))

        data = {
            'dict': {
                'src_char2idx': src_char2idx,
                'src_idx2char': src_idx2char,
                'tgt_char2idx': tgt_char2idx,
                'tgt_idx2char': tgt_idx2char
            }
        }
        with open(Config.vocab_file, 'wb') as file:
            pickle.dump(data, file)

    # 加载训练集和验证集
    train = get_data(Config.train_translation_zh_filename, Config.train_translation_en_filename)
    valid = get_data(Config.valid_translation_zh_filename, Config.valid_translation_en_filename)

    data = {
        'train': train,
        'valid': valid
    }
    # 这里面存的数据: 中文已映射成对应得id保存, 英文也已映射成id 并且加了其实标志和结束标志.他们都没有进行padding 只是有一个最大长度

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))

    with open(Config.data_file, 'wb') as file:
        pickle.dump(data, file)


