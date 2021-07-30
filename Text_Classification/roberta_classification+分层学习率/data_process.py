# -*- coding: utf-8 -*-
# @Time    : 2020/8/26 15:37
# @Author  : xiaolu
# @FileName: data_process.py
# @Software: PyCharm
import json
from transformers import BertTokenizer
from tqdm import tqdm
from pdb import set_trace


def convert_ids(path, save_path):
    # 1. 加载训练集
    data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = json.loads(line)
            data.append(line)

    result = []
    i = 0
    for items in tqdm(data):
        i += 1
        answer = items['answer']
        question = items['question']
        id = int(items['id'])
        label = items['yesno_answer']
        temp = tokenizer.encode_plus(question, answer, pad_to_max_length=True, max_length=240)   # token_type_ids, input_ids, attention_mask

        input_ids = temp['input_ids']
        token_type_ids = temp['token_type_ids']
        attention_mask = temp['attention_mask']
        label_id = label2id[label]

        result.append(
            {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels': label_id,
                'id': id
            }
        )
        if i < 10:
            print('input_ids:', input_ids)
            print('token_type_ids:', token_type_ids)
            print('attention_mask:', attention_mask)
            print('labels:', label_id)
            print('id:', id)

    with open(save_path, 'w', encoding='utf8') as fout:
        for i in result:
            fout.write(json.dumps(i, ensure_ascii=False) + '\n')

    print('len(features):', len(result))


if __name__ == '__main__':
    train_data_path = './work_dir/train.json'
    dev_data_path = './work_dir/dev.json'

    label2id = {'Yes': 0, 'No': 1, 'Depends': 2}
    id2label = {0: 'Yes', 1: 'No', 2: 'Depends'}

    tokenizer = BertTokenizer.from_pretrained('./roberta_large/vocab.txt', do_lower_case=False)

    # 处理训练集
    train_data_save_path = './work_dir/train_processed.work_dir'
    convert_ids(train_data_path, train_data_save_path)

    # 处理验证集
    dev_data_save_path = './work_dir/dev_processed.work_dir'
    convert_ids(dev_data_path, dev_data_save_path)





