# -*- encoding: utf-8 -*-
'''
@File    :   convert_token_ids.py
@Time    :   2020/11/04 13:39:13
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import os
import json
import pandas
from tqdm import tqdm
import gzip
import pickle
import random
from config import set_args


class RankExample(object):
    def __init__(self,
                 doc_id,
                 question_text,
                 context,
                 answer=None,
                 label=None,
                 keywords=None
                 ):
                 # keywords
        self.doc_id = doc_id
        self.question_text = question_text
        self.context = context
        self.answer = answer
        self.label = label
        self.keywords = keywords

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "doc_id: %s" % (str(self.doc_id))
        s += ", question_text: %s" % (self.question_text)
        s += ", context: %s" % (self.context)
        s += ", answer: %s" % (self.answer)
        s += ", label: %d" % (self.label)
        s += ", keyword: {}".format(self.keywords)
        return s


class InputFeatures(object):
    def __init__(self, input_ids, bigram, trigram, seq_len, label):
        self.input_ids = input_ids
        self.bigram = bigram
        self.trigram = trigram
        self.seq_len = seq_len
        self.label = label
    
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "input_ids: %s" % (str(self.input_ids))
        s += ", bigram: %s" % (self.bigram)
        s += ", trigram: %s" % (self.trigram)
        s += ", seq_len: %s" % (self.seq_len)
        s += ", label: %d" % (self.label)
        return s


def build_vocab(train_examples, dev_examples, tokenizer, max_size, min_freq):
    vocab_dic = {}
    for t in tqdm(train_examples):
        question = t.question_text
        context = t.context
        keywords = ','.join(t.keywords)
        text = ''.join([question, context, keywords])
        for word in tokenizer(text):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
        
    for d in tqdm(dev_examples):
        question = t.question_text
        context = t.context
        keywords = ','.join(t.keywords)
        text = ''.join([question, context, keywords])
        for word in tokenizer(text):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic)+1, SEP: len(vocab_dic)+2})
    return vocab_dic


def biGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets


def triGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    t2 = sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets


def convert_examples_to_features(args, examples, vocab):
    features = []
    for e in tqdm(examples):
        words_line = []
        question = e.question_text
        context = e.context
        keywords = ','.join(e.keywords)
        token = []
        token.extend(tokenizer(question))   # 问题加入
        token.append(SEP)   # 加入特殊字符  将问题和文章隔开 让模型去学习
        token.extend(tokenizer(keywords + context))  # 关键词与文章加入
        seq_len = len(token)
        if args.pad_size:
            if len(token) < args.pad_size:
                token.extend([PAD] * (args.pad_size - len(token)))
            else:
                token = token[:args.pad_size]
                seq_len = args.pad_size
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))
        
        # fasttext ngram
        buckets = args.n_gram_vocab
        bigram = []
        trigram = []
        for i in range(args.pad_size):
            bigram.append(biGramHash(words_line, i, buckets))
            trigram.append(triGramHash(words_line, i, buckets))
        features.append(InputFeatures(input_ids=words_line, bigram=bigram, trigram=trigram, seq_len=seq_len, label=int(e.label)))
    return features


if __name__ == '__main__':
    args = set_args()
    # 训练集
    with gzip.open('./mix_data/train_examples.pkl.gz', 'rb') as f:
        train_examples = pickle.load(f)   

    # 测试集
    with gzip.open('./mix_data/dev_examples.pkl.gz', 'rb') as f:
        dev_examples = pickle.load(f)
    
    # 建立分词器
    if args.is_word:
        # 如果你分好词了  可以将args.is_word置为True
        tokenizer = lambda x: x.split(' ')   # 词级别
    else:
        tokenizer = lambda x: [y for y in x]   # 字符级别
    
    min_freq=1
    MAX_VOCAB_SIZE = 10000
    UNK, PAD, SEP = '<UNK>', '<PAD>', '<SEP>'

    # 是否已构建好词表
    print('load to vocab...')
    vocab_path = './mix_data/vocab.pkl'
    if os.path.exists(vocab_path):
        vocab = pickle.load(open(vocab_path, 'rb'))
    else:
        # 建立词表
        vocab = build_vocab(train_examples, dev_examples, tokenizer, MAX_VOCAB_SIZE, min_freq)
        pickle.dump(vocab, open(vocab_path, 'wb'))
    print('loaded vocab finish!')

    print(len(vocab))   # 6713
    exit()
    # print(vocab)
    print('正在处理训练集...')
    train_features = convert_examples_to_features(args, train_examples, vocab)
    with gzip.open('./mix_data/train_features.pkl.gz', 'wb') as fout:
        pickle.dump(train_features, fout)

    print('正在处理验证集...')
    dev_features = convert_examples_to_features(args, dev_examples, vocab)
    with gzip.open('./mix_data/dev_features.pkl.gz', 'wb') as fout:
        pickle.dump(dev_features, fout)
    