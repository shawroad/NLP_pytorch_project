# -*- coding: utf-8 -*-
# @Time    : 2020/9/17 18:22
# @Author  : xiaolu
# @FileName: data_process.py
# @Software: PyCharm
import json
from tqdm import tqdm
import random
import gzip
import pickle
from fuzzywuzzy import fuzz


class RankExample(object):
    def __init__(self,
                 doc_id=None,
                 question_text=None,
                 question_type=None,
                 context=None,
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


def find_answer_span_text(text, ans):
    # temp_text, temp_ans
    '''
    通过答案将文章进行精简
    :param example:
    :return:
    '''
    answer = ans
    doc_tokens = text

    # 找出答案的起始和结束位置
    flag = doc_tokens.find(answer)
    if flag == -1:
        return doc_tokens

    start, end = flag, flag + len(answer)
    while end - start < 440:
        if start > 0:
            start -= 1
        if end < len(doc_tokens):
            end += 1
        if start == 0 and end == len(doc_tokens):
            break
    doc_tokens = doc_tokens[start: end]
    return doc_tokens


def get_neg_sample(examples):
    for example_base in tqdm(examples):
        neg_context_id = example_base.neg_context_id

        for example in examples:
            if example.doc_id == neg_context_id:
                # 找到负id对应的负样本
                res = example.context[:]
                if len(res) > 440:
                    # 我们采用截断的方式
                    temp_text = res
                    temp_ans = example.answer
                    res = find_answer_span_text(temp_text, temp_ans)
                example_base.neg_context = res
                break

    # 上面的每个负样本通过答案把他缩减到答案附近
    # 接下来对每个正样本做   这和上面的先后顺序不能搞乱
    result = []
    for example in examples:
        temp_text = example.context
        temp_ans = example.answer
        res = find_answer_span_text(temp_text, temp_ans)
        example.context = res
        result.append(example)
    return result


def sampling(examples, max_doc_id):
    result = []
    for example in examples:
        doc_id = example.doc_id
        sample_id = set()
        while len(sample_id) < 1:
            temp_id = random.randint(0, max_doc_id)
            if temp_id == doc_id:
                continue
            sample_id.add(temp_id)
        sample_id = list(sample_id)
        example.neg_context_id = sample_id[0]
        result.append(example)
    return result


if __name__ == '__main__':
    temp = json.load(open('./data/all_dev_train_6_23.json', 'r', encoding='utf8'))

    type2zh = json.load(open('./data/type2zh.json', 'r', encoding='utf8'))
    data = temp['data']
    doc_id = 0
    examples = []
    for corpus in tqdm(data):
        for item in corpus['paragraphs']:
            context = item['context'].replace(' ', '')  # 文章
            questions = item['qas']  # 问题列表
            if len(questions) == 0:
                continue

            # 取出第一个问题
            question = questions[0]
            question_text = question['question'].replace(' ', '')

            # 问题类型
            if 'question_intention' not in question:
                question_type = '其他型'
            else:
                question_type = type2zh.get(question['question_intention'], '其他型')

            # 答案
            if len(question['answers']) == 0:
                continue
            else:
                answer = question['answers'][0]['text'].replace(' ', '')

            examples.append(RankExample(doc_id=doc_id,
                                        question_text=question_text,
                                        question_type=question_type,
                                        context=context,
                                        answer=answer,
                                        label=1))
            doc_id += 1

    # 对每个样本采一个负样本
    examples = sampling(examples, len(examples)-1)

    # 加下来  把每个样本的负样本id转为负样本 添加进来
    examples = get_neg_sample(examples)
    # print(len(examples))   # 18381
    n = len(examples)

    data = json.load(open('./train_50_select.json', 'r', encoding='utf8'))

    for item in data:
        question = item['question']
        answer = item['answer']

        doc_id += 1
        pos_paragraph = item['pos_paragraph']
        examples.append(RankExample(doc_id=doc_id,
                                    question_text=question,
                                    context=pos_paragraph,
                                    answer=answer,
                                    label=1))
        doc_id += 1
        neg_paragraph = item['neg_paragraph']
        examples.append(RankExample(doc_id=doc_id,
                                    question_text=question,
                                    context=neg_paragraph,
                                    answer=answer,
                                    label=0))

    # data = json.load(open('./extract_paragraph.json', 'r', encoding='utf8'))
    # for item in tqdm(data):
    #     question = item['question']
    #     answer = item['answer']
    #     paragraphs = item['related_doc']
    #     for paragraph in paragraphs:
    #         n += 1
    #         body = paragraph['body']
    #         if fuzz.partial_ratio(answer, body) > 80:
    #             examples.append(RankExample(doc_id=doc_id,
    #                                         question_text=question,
    #                                         context=body,
    #                                         answer=answer,
    #                                         label=1))
    #         else:
    #             examples.append(RankExample(doc_id=doc_id,
    #                                         question_text=question,
    #                                         context=body,
    #                                         answer=answer,
    #                                         label=0))
    print(len(examples))   #
    # 保存
    with gzip.open('./data_vague/examples.pkl.gz', 'wb') as fout:
        pickle.dump(examples, fout)

