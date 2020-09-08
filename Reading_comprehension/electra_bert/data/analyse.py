# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 15:23
# @Author  : xiaolu
# @FileName: analyse.py
# @Software: PyCharm
import json
import numpy as np


if __name__ == '__main__':
    data = json.load(open('./train.json', 'r', encoding='utf8'))
    print(len(data))
    lengths = []
    for item in data:
        temp = ''.join(item['context'][0][1])
        lengths.append(len(temp))
    print('最大长度:', max(lengths))
    print('最短长度:', min(lengths))
    print('平均长度:', np.mean(lengths))

    c = 0
    for i in lengths:
        if i > 490:
            c += 1
    print(c/len(data))

    # 分析答案
    lengths = []
    for item in data:
        temp = item['question']
        if len(temp) == 0:
            print(item)
        lengths.append(len(temp))
    print('最大长度:', max(lengths))
    print('最短长度:', min(lengths))
    print('平均长度:', np.mean(lengths))

    # 分析答案
    lengths = []
    for item in data:
        temp = item['answer']
        if len(temp) == 0:
            print(item)
        lengths.append(len(temp))
    print('最大长度:', max(lengths))
    print('最短长度:', min(lengths))
    print('平均长度:', np.mean(lengths))







