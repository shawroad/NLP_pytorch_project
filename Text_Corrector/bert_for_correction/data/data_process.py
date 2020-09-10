# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 17:50
# @Author  : xiaolu
# @FileName: data_process.py
# @Software: PyCharm
import json
import re


if __name__ == '__main__':
    with open('40000.json', 'r', encoding='utf8') as f:

        lines = f.readlines()
        data = []
        for line in lines:
            line = json.loads(line)
            context = line['body']
            context = context.replace('\n', '')
            data.append(context)

    # with open('train.data', 'w', encoding='utf8') as f:
    #     f.write('\n'.join(data))

    # 直接将每段话 按句子切分
    result = []
    for d in data:
        d = re.split('[?？！!。.]', d)
        d = '\n'.join(d)
        result.append(d)
    with open('train.data', 'w', encoding='utf8') as f:
        f.write('\n'.join(result))
















