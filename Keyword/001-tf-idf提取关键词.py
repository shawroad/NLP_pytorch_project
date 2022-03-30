"""
@file   : 001-tf-idf提取关键词.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-30
"""
import jieba.analyse
import jieba.posseg

if __name__ == '__main__':
    with open('./news.txt', 'r', encoding='utf8') as f:
        text = f.read()

    # jieba是针对单个文本中  按句子为单位  计算出每个词的idf   tf等
    # ‘ns’, ‘n’, ‘vn’, ‘v’，提取地名、名词、动名词、动词
    res = jieba.analyse.extract_tags(text, topK=20, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v', 'nr', 'nt'))
    print(res)