"""
# -*- coding: utf-8 -*-
# @File    : data_process.py
# @Time    : 2021/1/26 2:18 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import json
import random
import gzip
import pickle
from tqdm import tqdm
from transformers import BertTokenizer


class RankExample(object):
    def __init__(self,
                 doc_id=None,
                 question_text=None,
                 context=None,
                 label=None,
                 ):
        self.doc_id = doc_id
        self.question_text = question_text
        self.context = context
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "doc_id: %s" % (str(self.doc_id))
        s += ", question_text: %s" % (self.question_text)
        s += ", context: %s" % (self.context)
        s += ", label: %d" % (self.label)
        return s


class RankFeature:
    def __init__(self, doc_id, input_ids, input_mask, segment_ids, label):
        self.doc_id = doc_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "doc_id: %s" % (str(self.doc_id))
        s += ", input_ids: %s" % (self.input_ids)
        s += ", input_mask: %s" % (self.input_mask)
        s += ", segment_ids: %s" % (self.segment_ids)
        s += ", label: %d" % (self.label)
        return s


def convert_examples(data):
    i = 0
    neg_examples = []
    pos_examples = []
    for d in tqdm(data):
        ques = d['question']
        answer = d['answer']
        items = d['related_doc']
        for item in items:
            i += 1
            context = item['body']
            keywords = ','.join(item['keywords'])
            title = item['title']
            context = keywords + title + context

            if context.find(answer) != -1:
                label = 1
                example = RankExample(doc_id=i, question_text=ques, context=context, label=label)
                pos_examples.append(example)
            else:
                label = 0
                example = RankExample(doc_id=i, question_text=ques, context=context, label=label)
                neg_examples.append(example)
    # 对负样本进行采样  负样本和正样本的比值为1: 0.7
    neg_examples = neg_examples[: int(len(pos_examples) * 0.7)]
    examples = []
    examples.extend(neg_examples)
    examples.extend(pos_examples)
    random.shuffle(examples)
    return examples


def convert_features(tokenizer, examples):
    features = []
    for item in tqdm(examples):
        input_id = []
        input_mask = []
        segment_id = []

        # 1. 加入CLS
        input_id.append(tokenizer.cls_token_id)

        # 2. 将问题编码 然后加入
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item.question_text))
        input_id.extend(ids)
        segment_id.extend([0]*len(input_id))

        # 3. 加入sep
        input_id.append(tokenizer.sep_token_id)

        # 4. 加入文章
        ids_ = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item.context))
        input_id.extend(ids_)
        segment_id.extend([1] * (len(input_id) - len(segment_id)))

        # 5. 进行padding
        if len(input_id) >= 512:
            input_id = input_id[:511]
            segment_id = segment_id[:511]
            # 长度大于等于512 截成511 加入sep
            input_id.append(tokenizer.sep_token_id)
            segment_id.append(1)
            input_mask.extend([1] * 512)
        else:
            # 加入sep
            input_id.append(tokenizer.sep_token_id)
            input_mask.extend([1]*len(input_id))
            segment_id.append(1)

            # 填充
            input_id.extend([0] * (512 - len(input_id)))
            input_mask.extend([0] * (512 - len(input_mask)))
            segment_id.extend([0] * (512 - len(segment_id)))

        assert len(input_id) == len(input_mask) and len(input_id) == len(segment_id)

        feature = RankFeature(doc_id=item.doc_id, input_ids=input_id, input_mask=input_mask, segment_ids=segment_id, label=int(item.label))
        features.append(feature)
    return features


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('./roberta_pretrain/vocab.txt')

    train_data = json.load(open('./data/mini_data/train_mini.json', 'r', encoding='utf8'))
    dev_data = json.load(open('./data/mini_data/dev_mini.json', 'r', encoding='utf8'))

    # 整理成examples
    print('整理example  ing...')
    train_examples = convert_examples(train_data)
    with gzip.open('./data/mini_data/train_examples.pkl.gz', 'wb') as fout:
        pickle.dump(train_examples, fout)

    dev_examples = convert_examples(dev_data)
    with gzip.open('./data/mini_data/dev_examples.pkl.gz', 'wb') as fout:
        pickle.dump(dev_examples, fout)

    # 转id
    print('整理feature  ing...')
    train_features = convert_features(tokenizer, train_examples)
    with gzip.open('./data/mini_data/train_features.pkl.gz', 'wb') as fout:
        pickle.dump(train_features, fout)

    dev_features = convert_features(tokenizer, dev_examples)
    with gzip.open('./data/mini_data/dev_features.pkl.gz', 'wb') as fout:
        pickle.dump(dev_features, fout)

