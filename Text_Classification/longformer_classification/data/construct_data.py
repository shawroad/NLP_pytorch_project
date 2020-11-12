# -*- encoding: utf-8 -*-
'''
@File    :   construct_data.py
@Time    :   2020/11/04 13:49:38
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import gzip
import pickle
import json
import jieba
from tqdm import tqdm
import random


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


def construct(data):
    doc_id = 0
    examples = []
    pos_sample = 0
    neg_sample = 0
    for item in tqdm(data):
        question = item['question']
        answer = item['answer']
        related_doc = item['related_doc']
        if len(related_doc) == 0:
            continue
        for doc in related_doc:
            doc_id += 1
            text = doc['body']
            keywords = doc['keywords']
            if text.find(answer) != -1:
                pos_sample += 1
                examples.append(RankExample(doc_id=doc_id, question_text=question, context=text, answer=answer, label=1, keywords=keywords))
            else:
                neg_sample += 1
                examples.append(RankExample(doc_id=doc_id, question_text=question, context=text, answer=answer, label=0, keywords=keywords))
    print('正样本个数:', pos_sample)   # 48611     12324
    print('负样本个数:', neg_sample)   # 692525      170526
    # 训练集  正:负=48611:692525
    # 验证集  正:负=12324:170526
    return examples



if __name__ == '__main__':
    # 加载全部数据
    train_data = json.load(open('../extract_data/train_policy.json', 'r', encoding='utf8'))
    dev_data = json.load(open('../extract_data/dev_policy.json', 'r', encoding='utf8'))
    
    train_examples = construct(train_data)
    with gzip.open('./train_examples.pkl.gz', 'wb') as fout:
        pickle.dump(train_examples, fout)

    dev_examples = construct(dev_data)
    with gzip.open('./dev_examples.pkl.gz', 'wb') as fout:
        pickle.dump(dev_examples, fout)

