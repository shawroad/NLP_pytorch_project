# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 16:12
# @Author  : xiaolu
# @FileName: step2_data_process.py
# @Software: PyCharm
import gzip
import pickle
from tqdm import tqdm


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


def find_neg_sample_sub_ans_region(examples):
    # 将每条数据中按照负样本的id找到负样本  并给文章瘦身
    for example_base in tqdm(examples):
        negative_doc_id = example_base.negative_doc_id
        temp_id = negative_doc_id[:]   # 得到当前样本的 负样本的id
        # print(temp_id)
        negative_sample = []
        for example in examples:
            if example.doc_id in temp_id:
                temp_id.remove(example.doc_id)
                res = example.doc_tokens
                if len(res) > 440:
                    temp_text = res
                    temp_ans = example.answer
                    res = find_answer_span_text(temp_text, temp_ans)
                negative_sample.append(res)

        example_base.negative_doc = negative_sample

        # print(example_base)
        # print(len(example_base.negative_doc))
        # print(len(example_base.negative_doc_id))

    # 上面的每个负样本通过答案把他缩减到答案附近
    # 接下来对每个正样本做   这和上面的先后顺序不能搞乱
    result = []
    for example in examples:
        temp_text = example.doc_tokens
        temp_ans = example.answer
        res = find_answer_span_text(temp_text, temp_ans)
        example.doc_tokens = res
        result.append(example)
    return result


if __name__ == '__main__':
    with gzip.open('./data/examples.pkl.gz', 'rb') as f:
        examples = pickle.load(f)

    result = find_neg_sample_sub_ans_region(examples)
    # 保存
    with gzip.open('./data/examples2.pkl.gz', 'wb') as fout:
        pickle.dump(result, fout)



