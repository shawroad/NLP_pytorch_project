# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 17:28
# @Author  : xiaolu
# @FileName: data_process.py
# @Software: PyCharm

import codecs
import re
import pandas as pd
import numpy as np
import collections


def originHandle():
    with open('./renmin.txt', 'r', encoding='utf8') as inp, open('./renmin2.txt', 'w', encoding='utf8') as outp:
        for line in inp.readlines():
            line = line.split('  ')
            i = 1
            while i < len(line) - 1:
                if line[i][0] == '[':
                    # 地名 原始语料中多个实体组成的地名 用中括号括起来的, 我们要把括号去掉，然后把多个实体连接起来
                    outp.write(line[i].split('/')[0][1:])
                    i += 1
                    while i < len(line) - 1 and line[i].find(']') == -1:
                        if line[i] != '':
                            outp.write(line[i].split('/')[0])
                        i += 1
                    outp.write(line[i].split('/')[0].strip() + '/' + line[i].split('/')[1][-2:] + ' ')

                elif line[i].split('/')[1] == 'nr':
                    # 处理人名  也是将分开的人名合并成一个整体
                    word = line[i].split('/')[0]
                    i += 1
                    if i < len(line) - 1 and line[i].split('/')[1] == 'nr':
                        outp.write(word + line[i].split('/')[0] + '/nr ')
                    else:
                        outp.write(word + '/nr ')
                        continue
                else:
                    outp.write(line[i] + ' ')
                i += 1
            outp.write('\n')


def originHandle2():
    with open('./renmin2.txt', 'r', encoding='utf8') as inp, open('./renmin3.txt', 'w', encoding='utf8') as outp:
        for line in inp.readlines():
            line = line.split(' ')
            i = 0
            while i < len(line) - 1:
                if line[i] == '':
                    i += 1
                    continue
                word = line[i].split('/')[0]   # 词/词性
                tag = line[i].split('/')[1]
                if tag == 'nr' or tag == 'ns' or tag == 'nt':
                    # nt组织之类的名字,  ns为地名, nr为人名
                    outp.write(word[0] + '/B_' + tag + " ")
                    for j in word[1: len(word) - 1]:
                        if j != ' ':
                            outp.write(j + '/M_' + tag + " ")
                    outp.write(word[-1] + '/E_' + tag + " ")
                else:
                    for wor in word:
                        outp.write(wor + '/O ')
                i += 1
            outp.write('\n')


def sentence2split():
    with open('./renmin3.txt', 'r', encoding='utf8') as inp, open('./renmin4.txt', 'w', encoding='utf8') as outp:
        texts = inp.read()
        sentences = re.split('[，。！？、‘’“”:]/[O]', texts)
        for sentence in sentences:
            if sentence != " ":
                outp.write(sentence.strip() + '\n')


def data2pkl():
    datas = list()
    labels = list()
    tags = set()
    tags.add('')
    input_data = open('renmin4.txt', 'r', encoding='utf8')
    for line in input_data.readlines():
        line = line.split()
        linedata = []
        linelabel = []
        numNotO = 0
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])   # 标签集合
            if word[1] != 'O':
                numNotO += 1   # 加这个if  就是防止当前这句话中没有我们需要进行标注的实体 全是O的话 直接pass掉 没必要训练

        if numNotO != 0:
            datas.append(linedata)
            labels.append(linelabel)

    input_data.close()

    # print(len(datas))   # 37924
    # print(len(labels))  # 37924
    # print(datas[0])   # ['中', '共', '中', '央', '总', '书', '记']
    # print(labels[0])   # ['B_nt', 'M_nt', 'M_nt', 'E_nt', 'O', 'O', 'O']
    # print(tags)    # {'', 'M_nt', 'M_nr', 'E_nr', 'B_nr', 'M_ns', 'E_nt', 'O', 'B_nt', 'B_ns', 'E_ns'}

    def flat_gen(x):
        def iselement(e):
            return not(isinstance(e, collections.Iterable) and not isinstance(e, str))
        for el in x:
            if iselement(el):
                yield el
            else:
                yield from flat_gen(el)

    all_words = [i for i in flat_gen(datas)]
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words) + 1)

    tags = [i for i in tags]
    tag_ids = range(len(tags))

    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)

    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)

    word2id["unknow"] = len(word2id) + 1
    id2word[len(word2id)] = "unknow"

    max_len = 60

    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
    df_data['x'] = df_data['words'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=43)

    import pickle
    with open('./renmindata.pkl', 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
        pickle.dump(x_valid, outp)
        pickle.dump(y_valid, outp)
    print('** Finished saving the data.')


if __name__ == '__main__':
    # originHandle()   # 将原始语料进行简单处理
    # originHandle2()   # 转为标记语言

    # sentence2split()    # 将所有段落分成一条一条的短句子

    data2pkl()   # 将数据集放在pkl文件中

