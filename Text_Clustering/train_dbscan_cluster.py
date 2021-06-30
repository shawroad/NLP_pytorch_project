"""
@file   : train_dbscan_cluster.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-30
"""
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class DBSCANClustering:
    def __init__(self, stopwords_path=None):
        self.stopwords = self.load_stopwords(stopwords_path)
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def load_stopwords(self, stopwords=None):
        # 加载停用词
        if stopwords:
            with open(stopwords, 'r', encoding='utf8') as f:
                return [line.strip() for line in f.readlines()]
        else:
            return []

    def preprocess_data(self, corpus_path):
        # 预处理数据
        # 分词 + 取出停用词
        corpus = []
        with open(corpus_path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                corpus.append(' '.join([word for word in jieba.lcut(line.strip()) if word not in self.stopwords]))
        return corpus

    def pca(self, weights, n_components=2):
        # 降维
        pca = PCA(n_components=n_components)
        return pca.fit_transform(weights)

    def get_text_tfidf_matrix(self, corpus):
        """
        获取tfidf矩阵
        :param corpus:
        :return:
        """

        # print(self.vectorizer.fit_transform(corpus))
        '''
          (0, 3836)	3    第一条语料中第一个词 的索引为3836  词频为3
          (0, 1750)	3    第一条语料中第二个词 的索引为1750  词频为3
          (0, 1499)	1    第一条语料中第三个词 的索引为1499  词频为1
          (0, 219)	4    ...
          (0, 2355)	1    ...
          (1, 2115)	7    第二条语料中第一个词 的索引为2115  词频为7
          (1, 1360)	4    依次类推
        '''

        # 得到每条语料中每个词的tfidf值
        tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))

        # 获取tfidf矩阵中权重
        weights = tfidf.toarray()
        return weights

    def dbscan(self, corpus_path, eps=0.1, min_samples=3, fig=False):
        '''
        :param corpus_path: 数据 一行一条数据
        :param eps: dbscan的半径参数
        :param min_samples: dbscan中半径内最小样本数
        :param fig: 是否对降维数据画图
        :return:
        '''
        corpus = self.preprocess_data(corpus_path)

        weights = self.get_text_tfidf_matrix(corpus)

        pca_weights = self.pca(weights)

        clf = DBSCAN(eps=eps, min_samples=min_samples)

        y = clf.fit_predict(pca_weights)
        if fig:
            plt.scatter(pca_weights[:, 0], pca_weights[:, 1], c=y)
            plt.show()

        # 每个样本所属的簇
        result = {}
        for text_idx, label_idx in enumerate(y):
            if label_idx not in result:
                result[label_idx] = [text_idx]
            else:
                result[label_idx].append(text_idx)
        return result


if __name__ == '__main__':
    dbscan = DBSCANClustering(stopwords_path='./data/stop_words.txt')
    result = dbscan.dbscan('./data/test_data.txt', eps=0.05, min_samples=3)
    print(result)
