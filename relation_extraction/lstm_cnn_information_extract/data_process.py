"""
# -*- coding: utf-8 -*-
# @File    : data_process.py
# @Time    : 2020/12/8 4:59 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import json
from tqdm import tqdm

def process_relation():
    all_50_schemas = set()
    with open('./data/all_50_schemas', 'r') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            line = json.loads(line)
            all_50_schemas.add(line['predicate'])

    # 关系与id的映射
    id2predicate = {i + 1: j for i, j in enumerate(all_50_schemas)}  # 0表示终止类别
    predicate2id = {j: i for i, j in id2predicate.items()}

    with open('./data/all_50_schemas_me.json', 'w', encoding='utf8') as f:
        json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)


def load_data(path, save_path):
    data = []
    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            line = json.loads(line)
            data.append(
                {
                    'text': line['text'],
                    'spo_list': [(i['subject'], i['predicate'], i['object']) for i in line['spo_list']]
                }
            )
            for c in line['text']:
                chars[c] = chars.get(c, 0) + 1
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # 处理关系
    process_relation()

    chars = {}

    # 处理训练集
    train_path = './data/train_data.json'
    save_train_path = './data/train_data_me.json'
    load_data(train_path, save_train_path)
    print(len(chars))

    # 处理测试集
    dev_path = './data/dev_data.json'
    save_dev_path = './data/dev_data_me.json'
    load_data(dev_path, save_dev_path)
    print(len(chars))

    # 用chars构建词表
    min_count = 2
    with open('./data/all_chars_me.json', 'w', encoding='utf8') as f:
        chars = {i: j for i, j in chars.items() if j >= min_count}
        id2char = {i+2: j for i, j in enumerate(chars)} # padding: 0, unk: 1
        char2id = {j: i for i, j in id2char.items()}
        json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)

