"""
@file   : data_preprocess.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-04
"""
import jsonlines
import json


def data_porcess(path, desc):
    data_entailment = []
    data_contradiction = []
    with open(path, 'r+', encoding='utf8') as f:
        for item in jsonlines.Reader(f):
            if item['gold_label'] == 'entailment':   # 蕴含关系
                data_entailment.append(item)
            elif item['gold_label'] == 'contradiction':
                data_contradiction.append(item)
    data_entailment = sorted(data_entailment, key=lambda x: x['sentence1'])
    data_contradiction = sorted(data_contradiction, key=lambda x: x['sentence1'])

    process = []
    i, j = 0, 0
    while i < len(data_entailment):
        origin = data_entailment[i]['sentence1']   # 从蕴含样本中取出第一句话
        for index in range(j, len(data_contradiction)):
            if data_entailment[i]['sentence1'] == data_contradiction[index]['sentence1']:
                # 从矛盾数据中找出第一句话也是origin的语料
                process.append({'origin': origin, 'entailment': data_entailment[i]['sentence2'], 'contradiction': data_contradiction[index]['sentence2']})
                j = index + 1
                break
        while i < len(data_entailment) and data_entailment[i]['sentence1'] == origin:
            i += 1

    with open(desc + '_proceed.txt', 'w') as f:
        for d in process:
            f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    train_file = './cnsd_snli_v1.0.train.jsonl'
    test_file = './cnsd_snli_v1.0.test.jsonl'
    dev_file = './cnsd_snli_v1.0.dev.jsonl'

    print('正在处理训练集ing...')
    data_porcess(train_file, desc='train')

    print('正在处理测试集ing...')
    data_porcess(test_file, desc='test')

    print('正在处理验证集ing...')
    data_porcess(dev_file, desc='dev')