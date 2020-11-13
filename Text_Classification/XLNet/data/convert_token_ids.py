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
from transformers import BertTokenizer


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
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


# question, context, keywords, label
def tokenizer_encoder(question, context, keywords, label):
    input_id, attention_mask, segment_id = [], [], []

    # 1. add cls
    input_id.append(tokenizer.convert_tokens_to_ids('[CLS]'))
    segment_id.append(0)

    # 2. add question
    if len(question) > 0:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question))
        input_id.extend(ids)
        segment_id.extend([0] * len(ids))

    # 3. add sep
    input_id.append(tokenizer.convert_tokens_to_ids('[SEP]'))
    segment_id.append(0)

    # add keywords
    if len(keywords) > 0:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(keywords))
        input_id.extend(ids)
        segment_id.extend([1] * len(ids))

    # add sep
    input_id.append(tokenizer.convert_tokens_to_ids('[SEP]'))
    segment_id.append(1)

    # 4. add context
    if len(context) > 0:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
        input_id.extend(ids)
        segment_id.extend([1] * len(ids))

    # 5. add sep
    input_id.append(tokenizer.convert_tokens_to_ids('[SEP]'))
    segment_id.append(1)
    # print(input_id)
    # print(segment_id)
    # print(len(input_id))
    # print(len(segment_id))
    assert len(input_id) == len(segment_id)

    if len(input_id) < max_seq_length:
        # 进行padding
        attention_mask = [1] * len(segment_id) + [0] * (max_seq_length - len(segment_id))
        segment_id = segment_id + [0] * (max_seq_length - len(segment_id))
        input_id = input_id + [0] * (max_seq_length - len(input_id))
    else:
        input_id = input_id[:max_seq_length]
        attention_mask = [1] * max_seq_length
        segment_id = segment_id[:max_seq_length]

    assert len(input_id) == len(attention_mask)
    assert len(input_id) == len(segment_id)

    return InputFeatures(input_ids=input_id, input_mask=attention_mask, segment_ids=segment_id, label_id=label)


def convert_examples_to_features(data, max_seq_length, tokenizer):
    features = []
    '''
        self.doc_id = doc_id
        self.question_text = question_text
        self.context = context
        self.answer = answer
        self.label = label
        self.keywords = keywords
    '''
    for item in tqdm(data):
        question = item.question_text
        context = item.context
        label = item.label
        keywords = ','.join(item.keywords)
        encode = tokenizer_encoder(question, context, keywords, label)
        features.append(encode)
    return features


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('../roberta_pretrain/vocab.txt')
    # 训练集
    with gzip.open('./train_examples.pkl.gz', 'rb') as f:
        train_examples = pickle.load(f)   
    max_seq_length = 1024
    train_features = convert_examples_to_features(train_examples, max_seq_length, tokenizer)
    with gzip.open('./train_features.pkl.gz', 'wb') as fout:
        pickle.dump(train_features, fout)

    # 测试集
    with gzip.open('./dev_examples.pkl.gz', 'rb') as f:
        dev_examples = pickle.load(f)
    max_seq_length = 1024
    dev_features = convert_examples_to_features(dev_examples, max_seq_length, tokenizer)

    with gzip.open('./dev_features.pkl.gz', 'wb') as fout:
        pickle.dump(dev_features, fout)
    