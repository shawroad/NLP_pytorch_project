# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 14:21
# @Author  : xiaolu
# @FileName: concat_data.py
# @Software: PyCharm
import json


if __name__ == '__main__':
    data_V1 = json.load(open('train20_V1.json', 'r', encoding='utf8'))
    data_V2 = json.load(open('train20_V2.json', 'r', encoding='utf8'))
    # print(len(data_V1))   # 1565
    # print(len(data_V2))   # 5054
    temp = data_V1[:1065]
    data_V1 = data_V1[1065:]
    data_V2.extend(temp)
    print(len(data_V1))  # 500
    print(len(data_V2))  # 6119

    fp = open('train.json', 'w', encoding='utf8')
    fp.write(json.dumps(data_V2, ensure_ascii=False))
    fp.close()

    fp = open('test.json', 'w', encoding='utf8')
    fp.write(json.dumps(data_V1, ensure_ascii=False))
    fp.close()




