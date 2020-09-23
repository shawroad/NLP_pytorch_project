# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 9:38
# @Author  : xiaolu
# @FileName: convert_to_id.py
# @Software: PyCharm
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
                 question_type,
                 context,
                 neg_context_id=None,
                 neg_context=None,
                 answer=None,
                 label=None,
                 ):
        self.doc_id = doc_id
        self.question_text = question_text
        self.question_type = question_type
        self.context = context
        self.neg_context_id = neg_context_id
        self.neg_context = neg_context
        self.answer = answer
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "doc_id: %s" % (str(self.doc_id))
        s += ", question_text: %s" % (self.question_text)
        s += ", question_type: %s" % (self.question_type)
        s += ", context: %s" % (self.context)
        s += ", neg_context_id: %d" % (self.neg_context_id)
        s += ", neg_context: %s" % (self.neg_context)
        s += ", answer: %s" % (self.answer)
        s += ", label: %d" % (self.label)
        return s


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "  input_ids: %s" % (str(self.input_ids))
        s += ", input_mask: %s" % (self.input_mask)
        s += ", segment_ids: %s" % (self.segment_ids)
        s += ", label: %s" % (self.label)
        return s


def tokenizer_encoder(corpus, max_len, label=None):
    input_id, attention_mask, segment_id = [], [], []

    # 1. add cls
    input_id.append(tokenizer.convert_tokens_to_ids('[CLS]'))

    # 2. text2id
    input_id.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(corpus)))
    if len(input_id) > max_len:
        input_id = input_id[:max_len]
        segment_id.extend([0] * len(input_id))
        attention_mask.extend([1] * len(input_id))
    else:
        attention_mask.extend([1] * len(input_id) + [0] * (max_len - len(input_id)))
        input_id = input_id + [0] * (max_len - len(input_id))
        segment_id.extend([0] * len(input_id))

    # 2. add sep
    input_id.append(tokenizer.convert_tokens_to_ids('[SEP]'))
    segment_id.append(0)
    attention_mask.append(0)
    # print(len(attention_mask))
    # print(len(input_id))
    # print(len(segment_id))

    assert len(attention_mask) == len(input_id) == len(segment_id)
    return InputFeatures(input_ids=input_id, input_mask=attention_mask, segment_ids=segment_id, label=label)


def convert_examples_to_features(data, max_seq_length, max_ques_length, tokenizer):
    features = []
    for item in tqdm(data):
        question = item.question_text
        context = item.context
        label = item.label
        # print(question)   # 问题
        # print(context)   # 文章
        # print(label)   # 0 or 1
        question_input = tokenizer_encoder(question, max_ques_length, label)
        context_input = tokenizer_encoder(context, max_seq_length)
        features.append([question_input, context_input])
    return features


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('./roberta_pretrain/vocab.txt')
    with gzip.open('./data/examples.pkl.gz', 'rb') as f:
        examples = pickle.load(f)

    # 将所有样本的token转为id
    max_seq_length = 440
    max_ques_length = 60
    features = convert_examples_to_features(examples, max_seq_length, max_ques_length, tokenizer)

    random.shuffle(features)

    # 从总体的样本中  分五百条给验证集
    dev_features = features[:500]
    train_features = features[500:]

    with gzip.open('./data/train_features.pkl.gz', 'wb') as fout:
        pickle.dump(train_features, fout)

    with gzip.open('./data/dev_features.pkl.gz', 'wb') as fout:
        pickle.dump(dev_features, fout)
