# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 14:36
# @Author  : xiaolu
# @FileName: step1_data_process.py
# @Software: PyCharm
import json
import gzip
import pickle
import random
import numpy as np


class RankExample(object):
    def __init__(self,
                 qas_id,
                 question_text,
                 question_type,
                 doc_tokens,
                 doc_id,
                 answer=None,
                 label=None,
                 negative_doc_id=None,
                 negative_doc=None,
                 ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.question_type = question_type
        self.doc_id = doc_id
        self.doc_tokens = doc_tokens
        self.answer = answer
        self.label = label
        self.negative_doc_id = negative_doc_id
        self.negative_doc = negative_doc

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", question_type: %s" % (self.question_type)
        s += ", doc_id: %s" % (str(self.doc_id))
        s += ", doc_tokens: %s" % (self.doc_tokens)
        s += ", answer: %s" % (self.answer)
        s += ", label: %d" % (self.label)
        s += ", negative_doc_id: {}".format(self.negative_doc_id)
        s += ", negative_doc: {}".format(self.negative_doc)
        return s


def task_sample(examples, max_doc_id):
    '''
    :param examples: 所有样本
    :param max_doc_id: 文章数  注意 不等于len(examples)
    :return:
    '''
    result = []
    for example in examples:
        doc_id = example.doc_id
        # 从0-max_doc_id抽九个文章id
        sample_id = set()
        while len(sample_id) < 9:
            temp_id = random.randint(1, max_doc_id)
            if temp_id == doc_id:
                continue
            if temp_id == 2021:
                continue
            if temp_id == 2023:
                continue

            sample_id.add(temp_id)
        sample_id = list(sample_id)
        example.negative_doc_id = sample_id
        result.append(example)
    return result


if __name__ == '__main__':
    all_data_path = './data/train.json'
    all_data = json.load(open(all_data_path, 'r', encoding='utf8'))
    # print(len(all_data['data']))    # 11855

    # 构造正负样本  用前2000的文本
    data = all_data['data'][:2000]

    # 加载问题类型
    path = '../data/type2zh.json'
    type2zh = json.load(open(path, 'r', encoding='utf8'))

    examples = []
    # 开始构造数据
    doc_id = 0
    for corpus in data:
        for item in corpus['paragraphs']:
            doc_id += 1
            context = item['context']   # 文章
            questions = item['qas']    # 问题列表
            for ques in questions:
                question_id = ques['id']
                question = ques['question']   # 问题

                # 问题类型
                if 'question_intention' not in ques:
                    question_type = '其他型'
                else:
                    question_type = type2zh.get(ques['question_intention'], '其他型')

                # 答案
                if len(ques['answers']) == 0:
                    continue
                else:
                    answer = ques['answers'][0]['text']

                examples.append(
                    RankExample(
                        qas_id=question_id,
                        question_text=question.replace(' ', ''),
                        question_type=question_type,
                        doc_id=doc_id,
                        doc_tokens=context.replace(' ', ''),
                        answer=answer.replace(' ', ''),
                        label=1
                    ))

    max_doc_id = doc_id
    # print(max_doc_id)   # 4746
    # 保存
    # with gzip.open('./data/examples.pkl.gz', 'wb') as fout:
    #     pickle.dump(examples, fout)

    # 对每个样本采九个负样本
    result = task_sample(examples, max_doc_id)
    # print(len(result))
    for i, res in enumerate(result):
        if i % 500 == 0:
            print(res)

    with gzip.open('./data/examples.pkl.gz', 'wb') as fout:
        pickle.dump(result, fout)























