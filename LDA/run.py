"""
@file   : run.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-18
"""
import re
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def top_words_data_frame(model: LatentDirichletAllocation,
                         tf_idf_vectorizer: TfidfVectorizer,
                         n_top_words: int) -> pd.DataFrame:
    rows = []
    feature_names = tf_idf_vectorizer.get_feature_names()
    for topic in model.components_:
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        rows.append(top_words)  # 每个主题下的词
    columns = [f'topic {i+1}' for i in range(n_top_words)]
    df = pd.DataFrame(rows, columns=columns)
    return df


def predict_to_data_frame(model: LatentDirichletAllocation, X: np.ndarray) -> pd.DataFrame:
    # 求出给定文档的主题概率分布矩阵
    matrix = model.transform(X)
    columns = [f'P(topic {i+1})' for i in range(len(model.components_))]
    df = pd.DataFrame(matrix, columns=columns)
    return df


if __name__ == '__main__':
    # 1. 清洗数据 并分词
    document_column_name = '回答内容'
    pattern = u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!\t"@#$%^&*\\-_=+a-zA-Z，。\n《》、？：；“”‘’｛｝【】（）…￥！—┄－]+'
    df = pd.read_csv('./answers.csv', encoding='utf-8-sig').drop_duplicates().rename(columns={document_column_name: 'text'})
    df['cut'] = df['text'].apply(lambda x: str(x)).apply(lambda x: re.sub(pattern, ' ', x)).apply(lambda x: " ".join(jieba.lcut(x)))

    # 2. 构建tf-idf
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf = tf_idf_vectorizer.fit_transform(df['cut'])
    # 特征词列表
    feature_names = tf_idf_vectorizer.get_feature_names()
    # 特征词 TF-IDF 矩阵
    matrix = tf_idf.toarray()
    feature_names_df = pd.DataFrame(matrix, columns=feature_names)
    # print(feature_names_df.head())

    # 3. LDA提取主题
    n_topics = 5   # 指定 lda 主题数  需要手动指定
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=50,
        learning_method='online',
        learning_offset=50.,
        random_state=0)

    # 核心，给 LDA 喂生成的 TF-IDF 矩阵
    lda.fit(tf_idf)

    n_top_words = 20   # 每个主题下前20个词
    top_words_df = top_words_data_frame(lda, tf_idf_vectorizer, n_top_words)
    top_words_df.to_csv('./top_vocab.csv', encoding='utf-8-sig', index=False)

    # 转 tf_idf 为数组，以便后面使用它来对文本主题概率分布进行计算
    X = tf_idf.toarray()

    # 计算完毕主题概率分布情况
    predict_df = predict_to_data_frame(lda, X)   # 获取每个样本所属主题的概率分布

    # 保存文本主题概率分布到 csv 文件中
    predict_df.to_csv('./result.csv', encoding='utf-8-sig', index=False)
