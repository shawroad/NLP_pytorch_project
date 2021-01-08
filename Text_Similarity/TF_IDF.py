"""
# -*- coding: utf-8 -*-
# @File    : TF_IDF.py
# @Time    : 2021/1/8 3:30 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import numpy as np
import jieba


class TF_IDF_Model:
    def __init__(self, documents_list):
        # 分好词的文本列表  这里有若干文本
        self.documents_list = documents_list

        # 文本总个数
        self.documents_number = len(documents_list)

        # 存储每个文本中每个词的词频
        self.tf = []

        # 存储每个词汇的逆文档频率
        self.idf = {}

        # 类初始化
        self.init()

    def init(self):
        # 这里的初始化  就是对已有的文本库计算tf和idf
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                # 存储每个文档中每个词的词频
                temp[word] = temp.get(word, 0) + 1 / len(document)
            self.tf.append(temp)

            for key in temp.keys():
                # 当前词在本篇文章出现过  那就+1
                df[key] = df.get(key, 0) + 1

        for key, value in df.items():
            # 每个词的逆文档频率
            self.idf[key] = np.log(self.documents_number / (value + 1))

    def get_score(self, index, query):
        '''
        :param index: 传进文本库的某个问题的索引
        :param query: 用户提问的问题
        :return: 用户提问问题与当前文本库中某个问题的相似性得分
        '''
        score = 0.0
        for q in query:
            # 遍历用户提问问题中每个词
            if q not in self.tf[index]:
                continue
            score += self.tf[index][q] * self.idf[q]   # 用户提问问题中每个词在本文中的tf和idf
        return score

    def get_documents_score(self, query):
        '''
        :param query: 用户提问的问题
        :return: 返回与文本库中每个问题的相似度得分
        '''
        # 计算当前问题与所有文本库中的问题 相似度
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
    # print(document_list)   # 对每个问题进行了分词
    tf_idf_model = TF_IDF_Model(document_list)
    # print(tf_idf_model.tf)   # [{词1: tf值, 词2: tf值, 词3: tf值 ...}, {第二篇文章}, {第三篇文章}...]
    # print(tf_idf_model.idf)  # {词1: idf值, 词2: idf值, 词3: idf值, 词4: idf值...}

    # 给出一个问题，检索文本库中哪个问题跟当前这个问题相似
    query = "走私了两万元，在法律上应该怎么量刑？"
    query = list(jieba.cut(query))
    scores = tf_idf_model.get_documents_score(query)
    print(scores)   # 得分列表
    # [0.0021669905358997106, 0.0256563880603619, 0.17167897852134476, 0.0013414703317, ...]
    max_value_index = np.argmax(scores)  # 最高分的索引位置  输出4
    print(''.join(document_list[max_value_index]))   # 输出: 有人走私两万元，怎么处置他？

