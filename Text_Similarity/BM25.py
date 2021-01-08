"""
# -*- coding: utf-8 -*-
# @File    : BM25.py
# @Time    : 2021/1/8 4:10 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import jieba
import numpy as np
from collections import Counter


class BM25_Model(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.5):
        # 标准问题库
        self.documents_list = documents_list

        # 标准问题库中问题个数
        self.documents_number = len(documents_list)

        # avg_documents_len 表示所有文本的平均长度
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number

        # f 用于存储每个文本中每个词的出现的次数  一个词在当前文本中的词频
        self.f = []

        # idf用于存储每个词汇的权重值
        self.idf = {}

        self.k1 = k1
        self.k2 = k2
        self.b = b

        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1

        for key, value in df.items():
            # 这个就是上述公式中的Wi
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)  # 统计用户输入问题的词频

        for q in query:
            if q not in self.f[index]:    # self.f[index]就是取出当前文本库中的一个问题
                # 如果当前用户问题中的某个词都不在这个问题中，则直接跳过
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                    self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                qf[q] * (self.k2 + 1) / (qf[q] + self.k2)
            )
        return score

    def get_documents_score(self, query):
        '''
        :param query: 用户输入的问题
        :return: 与文本库中每个问题的相似度得分
        '''
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list


if __name__ == '__main__':
    document_list = ["行政机关强行解除行政协议造成损失，如何索取赔偿？",
                     "借钱给朋友到期不还得什么时候可以起诉？怎么起诉？",
                     "我在微信上被骗了，请问被骗多少钱才可以立案？",
                     "公民对于选举委员会对选民的资格申诉的处理决定不服，能不能去法院起诉吗？",
                     "有人走私两万元，怎么处置他？",
                     "法律上餐具、饮具集中消毒服务单位的责任是不是对消毒餐具、饮具进行检验？"]

    document_list = [list(jieba.cut(doc)) for doc in document_list]

    bm25_model = BM25_Model(document_list)

    query = "走私了两万元，在法律上应该怎么量刑？"
    query = list(jieba.cut(query))

    scores = bm25_model.get_documents_score(query)
    print(scores)

    index = np.argmax(scores)
    print(''.join(document_list[index]))






