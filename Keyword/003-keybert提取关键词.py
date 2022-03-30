"""
@file   : 003-keybert提取关键词.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-30
"""
from keybert import KeyBERT
# pip install keybert
import jieba


if __name__ == '__main__':
    # keybert相当于是对每个词 和文本进行相关性计算
    model = KeyBERT('bert-base-chinese')

    with open('news.txt', 'r', encoding='utf8') as f:
        text = f.read()
    doc = " ".join(jieba.cut(text))
    keywords = model.extract_keywords(doc, keyphrase_ngram_range=(1,2),  top_n=20)
    print(keywords)
