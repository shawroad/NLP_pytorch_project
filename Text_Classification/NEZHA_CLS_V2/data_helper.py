"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-01-05
"""
import re
import torch
import pandas as pd
from torch.utils.data import Dataset


def to_list(row):
    return list(map(int, row.split(',')))


def clean_text(text):
    rule_url = re.compile(
        '(https?://)?(www\\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_+.~#?&/=]*)'
    )
    rule_legal = re.compile('[^\\[\\]@#a-zA-Z0-9\u4e00-\u9fa5]')
    rule_space = re.compile('\\s+')
    text = str(text).replace('\\n', ' ').replace('\n', ' ').strip()
    text = rule_url.sub(' ', text)
    text = rule_legal.sub(' ', text)
    text = rule_space.sub(' ', text)
    return text.strip()


def load_data(path):
    df = pd.read_csv(path)
    df = df.loc[:, ['ad', 'label']]   # 取出广告和标签
    df['ad'] = df['ad'].astype(str)
    df['label'] = df['label'].astype(str)  # 这里的label时onehot形式，如 ['0, 1, 0, 0, 0', '1, 0, 1, 0, 0']

    df.dropna(subset=['label', 'ad'], inplace=True)

    df.drop_duplicates(subset='ad', keep='first', inplace=True)  # 去重

    df.reset_index(drop=True, inplace=True)   # 重置索引

    df.loc[:, "label"] = df['label'].map(to_list)   # 从onehot的字符串形式转为列表形式
    df.loc[:, 'ad'] = df['ad'].map(clean_text)   # 清洗文本
    train_size = 0.95   # 分一些验证集

    train_df = df.sample(frac=train_size, random_state=200)
    val_df = df.drop(train_df.index).reset_index(drop=True)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.ad
        self.label = dataframe.label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.text[index],
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': self.label[index]
        }


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_fn(batch):
    # 按batch进行padding获取当前batch中最大长度
    # max_len = max([len(d['input_ids']) for d in batch])

    # 定一个全局的max_len
    max_len = 128

    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(item['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(item['token_type_ids'], max_len=max_len))
        labels.append(item['label'])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.float)
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids
