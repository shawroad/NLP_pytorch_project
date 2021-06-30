"""
@file   : train_LDA_cluster.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-30
"""
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LDAClustering:
    def __init__(self, stopwords_path=None, n_components=5, learning_method='batch', max_iter=10):
        stopwords = self.load_stopwords(stopwords_path)
        self.CountVector = CountVectorizer(stop_words=stopwords)
        self.n_components = n_components
        self.learning_method = learning_method
        self.max_iter = max_iter

    def load_stopwords(self, stopwords_path):
        '''
        加载停用词
        :param stopwords_path:
        :return:
        '''
        with open(stopwords_path, 'r', encoding='utf8') as f:
            return [line.strip() for line in f.readlines()]

    def pre_process_corpus(self, corpus_path):
        '''
        预处理语料
        :param corpus_path:
        :param stopwords_path:
        :return:
        '''
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = [' '.join(jieba.lcut(line.strip())) for line in f.readlines()]
        cntTf = self.CountVector.fit_transform(corpus)
        return cntTf

    def fmt_lda_result(self, lda_result):
        '''
        整理结果
        :param lda_result:
        :return:
        '''
        result = {}
        for doc_index, res in enumerate(lda_result):
            li_res = list(res)   # 每一行代表了属于每个主题的概率
            doc_label = li_res.index(max(li_res))
            if doc_label not in result:
                result[doc_label] = [doc_index]
            else:
                result[doc_label].append(doc_index)
        return result

    def lda(self, corpus_path):
        '''
        :param corpus_path: 语料
        :param n_components: 聚类的数目
        :param learning_method: 学习方法 'batch or online'
        :param max_iter: EM算法迭代次数
        :param stopwords_path: 停用词
        :return:
        '''
        cntTf = self.pre_process_corpus(corpus_path=corpus_path)
        lda = LatentDirichletAllocation(n_components=self.n_components, max_iter=self.max_iter, learning_method=self.learning_method)
        docres = lda.fit_transform(cntTf)
        return self.fmt_lda_result(docres)


if __name__ == '__main__':
    LDA = LDAClustering(
        stopwords_path='./data/stop_words.txt',
        n_components=5,
        learning_method='batch',
        max_iter=10
    )
    result = LDA.lda('./data/test_data.txt')
    print(result)
