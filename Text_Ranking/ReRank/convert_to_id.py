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
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def tokenizer_encoder(question, context, label):
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

    # 4. add context
    if len(context) > 0:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
        input_id.extend(ids)
        segment_id.extend([1] * len(ids))

    # 5. add sep
    input_id.append(tokenizer.convert_tokens_to_ids('[SEP]'))
    segment_id.append(0)
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
        attention_mask = [1] * 512
        segment_id = segment_id[:max_seq_length]

    assert len(input_id) == len(attention_mask)
    assert len(input_id) == len(segment_id)

    return InputFeatures(input_ids=input_id, input_mask=attention_mask, segment_ids=segment_id, label_id=label)


def convert_examples_to_features(data, max_seq_length, tokenizer):
    features = []
    '''
    RankExample(doc_id=doc_id,
                                        question_text=question_text,
                                        question_type=question_type,
                                        context=context,
                                        answer=answer,
                                        label=1))
                                        '''
    for item in tqdm(data):
        question = item.question_text
        context = item.context
        label = item.label
        # neg_context = item.neg_context
        encode = tokenizer_encoder(question, context, label)
        features.append(encode)
    return features


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('./roberta_pretrain/vocab.txt')
    with gzip.open('./data/examples.pkl.gz', 'rb') as f:
        examples = pickle.load(f)

    # 将所有样本的token转为id
    max_seq_length = 512
    features = convert_examples_to_features(examples, max_seq_length, tokenizer)
    random.shuffle(features)
    # 从总体的样本中  分五百条给验证集
    dev_features = features[:500]
    train_features = features[500:]

    with gzip.open('./data/train_features.pkl.gz', 'wb') as fout:
        pickle.dump(train_features, fout)

    with gzip.open('./data/dev_features.pkl.gz', 'wb') as fout:
        pickle.dump(dev_features, fout)
