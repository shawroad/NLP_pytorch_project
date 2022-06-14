"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-14
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_data(path):
    texts = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            texts.append(line)
    return texts


class RobertaDataSet(Dataset):
    def __init__(self, path, tokenizer):
        super(RobertaDataSet, self).__init__()
        self.data = load_data(path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in self.tokenizer.tokenize(self.data[item])]

        input_ids, lm_mask_label = self.random_masking(token_ids)

        # 前加CLS
        input_ids.insert(0, self.tokenizer.cls_token_id)
        lm_mask_label.insert(0, 0)

        # 后加SEP
        input_ids.append(self.tokenizer.sep_token_id)
        lm_mask_label.append(0)

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        return {"input_ids": input_ids, 'attention_mask': attention_mask,
                'token_type_ids': token_type_ids, 'lm_mask_label': lm_mask_label}

    def random_masking(self, token_ids):
        """
        对输入进行随机mask
        """
        rands = np.random.random(len(token_ids))   # 随机生成一个跟token_id一样维度的概率矩阵
        source, target = [], []
        for r, t in zip(rands, token_ids):
            # 以下都是在15%上操作的
            if r < 0.15 * 0.8:
                # 80% mask
                source.append(self.tokenizer.mask_token_id)
                target.append(t)

            elif r < 0.15 * 0.9:
                # 10% 不替换
                source.append(t)
                target.append(t)

            elif r < 0.15:
                # 10% 随机填一个词
                source.append(np.random.choice(self.tokenizer.vocab_size - 1) + 1)
                target.append(t)
            else:
                source.append(t)
                target.append(0)
        return source, target


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_fn(batch):
    max_len = max([len(d['input_ids']) for d in batch])
    if max_len > 512:
        max_len = 512
    # max_len = 256

    input_ids, attention_mask, token_type_ids, lm_mask_labels = [], [], [], []
    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(item['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(item['token_type_ids'], max_len=max_len))
        lm_mask_labels.append(pad_to_maxlen(item['lm_mask_label'], max_len=max_len))

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_lm_mask_labels = torch.tensor(lm_mask_labels, dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids, all_lm_mask_labels
