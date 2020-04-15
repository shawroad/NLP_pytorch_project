"""

@file  : data_helper_.py

@author: xiaolu

@time  : 2020-04-14

"""
import pandas as pd
from transformers import BertTokenizer
import json
from tqdm import tqdm
import numpy as np
from config import Config


def convert_examples_to_features(data, tokenizer, max_seq_length, max_query_length):
    '''
    预处理数据
    :param data:
    :param tokenizer:
    :param max_seq_length:
    :param max_query_length:
    :return:
    '''
    max_len = 512
    context = data['context'].tolist()
    question = data['question'].tolist()
    answer = data['answer'].tolist()

    input_data = []
    input_len = []
    mask_data = []
    output_data = []

    for ctext, ques, ans in tqdm(zip(context, question, answer)):
        if len(ctext) > max_seq_length:
            ctext = ctext[:max_seq_length]

        if len(ques) > max_query_length:
            ques = ques[:max_query_length]

        tokens = []
        segment_ids = []

        tokens.append('[CLS]')
        segment_ids.append(0)

        # 把问题当做bert输入的前半部分   [CLS] 问题的id形式　[SEP] 文章的id形式 [SEP]
        for char_q in ques:
            tokens.append(char_q)
            segment_ids.append(0)

        tokens.append('[SEP]')
        segment_ids.append(0)

        for i in ctext:
            tokens.append(i)
            segment_ids.append(1)

        tokens.append('[SEP]')
        segment_ids.append(1)

        # 转id
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 加入mask
        input_mask = [1] * len(input_ids)

        item_len = len(input_ids)

        if len(input_ids) < max_len:
            input_ids.extend([0] * (max_len - len(input_ids)))

        if len(input_mask) < max_len:
            input_mask.extend([0] * (max_len - len(input_mask)))

        # 起始标志21130   结束标志21131
        ans = tokenizer.encode(ans)   # 编码后需要将[CLS], [SEP] 扔掉　这是我们的解码输出
        ans.pop(0)
        ans.pop(-1)

        output_ids = ans + [Config.EOS]

        if len(output_ids) < Config.ans_max_len:
            output_ids.extend([0] * (Config.ans_max_len - len(output_ids)))
        else:
            output_ids = output_ids[:Config.ans_max_len]

        input_data.append(input_ids)
        mask_data.append(input_mask)
        input_len.append(item_len)
        output_data.append(output_ids)

    # 按照原始输入数据的长短进行排序 从长到短
    length = np.array(input_len)
    ids = length.argsort()
    ids = ids.tolist()
    ids = list(reversed(ids))

    f_input = []
    f_mask = []
    f_input_l = []
    f_output = []
    for i in ids:
        f_input.append(input_data[i])
        f_mask.append(mask_data[i])
        f_input_l.append(input_len[i])
        f_output.append(output_data[i])

    data = {
        'input_data': f_input,
        'mask_data': f_mask,
        'input_len': f_input_l,
        'output_data': f_output
    }

    with open('./data/train.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('./albert_pretrain/vocab.txt')

    # ids,context,question,answer,start_position,end_position
    path = './data/correct_data.csv'
    data = pd.read_csv(path)
    convert_examples_to_features(data, tokenizer, max_seq_length=460, max_query_length=40)

