"""
@file   : train_kmeans_cluster.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-30
"""
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans


class KmeansClustering:
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

    def kmeans(self, corpus_path, n_clusters=5):
        '''
        文本聚类函数
        :param corpus_path: 语料  一行一个文本 id从零开始
        :param n_clusters: 簇的个数
        :return: {簇1:[文本1,文本2, ...], 簇2:[文本i, 文本i+1, ...], ...}
        '''
        corpus = self.preprocess_data(corpus_path)    # 分词+取出停用词

        weights = self.get_text_tfidf_matrix(corpus)
        # print(weights.shape)    # (25, 3863)    每条语料对应一个3863维向量 每个值对应其tfidf值

        clf = KMeans(n_clusters=n_clusters)

        # clf.fit(weights)
        # print('中心点:', clf.cluster_centers_)
        # print('score:', clf.inertia_)   # 用来评估簇的个数是否合适,距离越小说明簇分得越好,选取临界点的簇的个数

        y = clf.fit_predict(weights)

        # 每个样本所属的簇
        result = {}
        for text_idx, label_idx in enumerate(y):
            if label_idx not in result:
                result[label_idx] = [text_idx]
            else:
                result[label_idx].append(text_idx)
        return result


if __name__ == '__main__':
    K_means = KmeansClustering(stopwords_path='./data/stop_words.txt')
    result = K_means.kmeans('./data/test_data.txt', n_clusters=5)
    print(result)
