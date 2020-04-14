"""

@file  : data_helper.py

@author: xiaolu

@time  : 2020-04-09

"""
import json
import pandas as pd
from transformers import BertTokenizer


def convert_examples_to_features(data, tokenizer, max_seq_length, max_query_length):
    '''
    :param examples: 处理好的文本 以字典的格式给出
    :param tokenizer: bert分词器
    :param max_seq_length: 文章的长度
    :param max_query_length: 问题长度
    :return:
    '''
    context = data['context'].tolist()
    question = data['question'].tolist()
    answer = data['answer'].tolist()
    start = data['start_position'].tolist()

    end = []
    begin = []
    for a, s in zip(answer, start):
        s = float(s)
        s = int(s)
        begin.append(s)
        end.append(s + len(a))

    features = []
    id = 0
    for ctext, ques, ans, start_position, end_position in zip(context, question, answer, begin, end):
        id += 1
        # print(ctext)    # 你家住在哪？我家住在花园路五百七十号。你家的电话号码是多少？我家的电话号码是：二零七六 一五八四。
        # print(ques)    # 你家住在哪？
        # print(ans)   # 我家住在花园路五百七十号
        # print(ctext[start_position:end_position])   # 我家住在花园路五百七十号
        # print(start_position)   # 6
        # print(end_position)   # 18

        if len(ctext) > max_seq_length:
            ctext = ctext[:max_seq_length]

        if start_position > max_seq_length or end_position > max_seq_length:
            continue

        if len(ques) > max_query_length:
            question = question[:max_query_length]

        tokens = []
        segment_ids = []

        tokens.append('[CLS]')
        segment_ids.append(0)

        # 加入CLS　答案的起始和结束需要加1
        start_position = start_position + 1
        end_position = end_position + 1

        # 把问题当做bert输入的前半部分   [CLS] 问题的id形式　[SEP] 文章的id形式 [SEP]
        for char_q in ques:
            tokens.append(char_q)
            segment_ids.append(0)

            start_position = start_position + 1
            end_position = end_position + 1

        tokens.append('[SEP]')
        segment_ids.append(0)
        start_position = start_position + 1
        end_position = end_position + 1

        for i in ctext:
            tokens.append(i)
            segment_ids.append(1)

        tokens.append('[SEP]')
        segment_ids.append(1)

        # 转id
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 加入mask
        input_mask = [1] * len(input_ids)

        assert len(input_ids) == len(segment_ids)

        features.append(
            {
                'ids': id,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'start_position': start_position,
                'end_position': end_position,
            }
        )

    with open('./data/train.data', 'w', encoding='utf8') as fout:
        for feature in features:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')

    print('len(features):', len(features))
    return features


if __name__ == '__main__':
    # 1. 将文章转为id用的
    tokenizer = BertTokenizer.from_pretrained('./albert_pretrain/vocab.txt')

    # 2. 加载抽取的数据集
    data = pd.read_csv('./data/correct_data.csv')
    # print(data.shape[0])   # 13718

    # 将含有nan的行去除掉
    data = data.dropna(axis=0)
    # print(data.shape[0])   # 13575
    convert_examples_to_features(data, tokenizer, max_seq_length=460, max_query_length=40)

