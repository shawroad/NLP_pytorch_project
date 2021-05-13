"""
@file   : data_process.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-11
"""
'''建立词表 并将文字转为token序列'''
import os


def build_vocab():
    # 获取所有的语料 统计字
    text_path = 'data/corpus.txt'
    words = set()
    with open(text_path, 'r', encoding='utf8') as f:
        word = f.read(1)   # 每次都一个字符
        while word:
            if word == '\n' or word == '\r' or word == ' ':
                pass
            else:
                words.add(word)
            word = f.read(1)

    with open('./data/vocab.txt', "w+", encoding="utf-8") as f:
        f.write("[START] [SEQ] [UNK] [PAD] [END] ")
        f.write(" ".join(words))
        f.flush()


def convert_id():
    # 先读取字典
    with open('./data/vocab.txt', 'r', encoding='utf8') as f:
        dics = f.read().strip().split()
    # print(dics)

    with open('./data/corpus.txt', 'r', encoding='utf8') as f:
        indexs = ['0']   # 起始标志
        word = f.read(1)
        while word:
            if word == '\n' or word == '\r' or word == '\t' or ord(word) == 12288:
                indexs.append('1')   # 使用sep隔开
            elif word == ' ':
                indexs.append('3')   # 如果是空格 则整成pad
            else:
                try:
                    indexs.append(str(dics.index(word)))
                except:
                    indexs.append("2")   # 词表没有的  直接整成unk
            word = f.read(1)

    with open('./data/processed_data.txt', 'w', encoding='utf8') as f:
        f.write(' '.join(indexs))


if __name__ == '__main__':
    if not os.path.exists('./data/vocab.txt'):
        build_vocab()

    convert_id()

