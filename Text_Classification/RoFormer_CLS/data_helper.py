"""
@file   : utils.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-22
"""
import re
import torch
import pandas as pd
from torch.utils.data import Dataset


def to_list(row):
    return list(map(int, row['target'].split(',')))


def load_data(path, is_train=True):
    '''
    加载数据
    :return:
    '''
    df = pd.read_csv(path)
    df = df.loc[:, ['name_content', 'target']]   # 取出内容和标签
    df['name_content'] = df['name_content'].astype(str)
    df['target'] = df['target'].astype(str)
    df = df[df.target != "nan"]
    df.dropna(subset=["name_content", "target"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.loc[:, "target"] = df.apply(to_list, axis=1)
    train_size = 0.95
    train_df = df.sample(frac=train_size, random_state=200)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if not is_train:
        train_df = train_df.sample(frac=0.005, random_state=44)
    return train_df, val_df


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.comment_text = dataframe.name_content
        self.targets = dataframe.target

    def __len__(self):
        return len(self.comment_text)

    def text_normal_l1(self, text):
        # 对数据进行简单清洗
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

    def __getitem__(self, index):
        comment_text = self.text_normal_l1(self.comment_text[index])
        inputs = self.tokenizer.encode_plus(
            text=comment_text,
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
            'targets': self.targets[index]
        }
