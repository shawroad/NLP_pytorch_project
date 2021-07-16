"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-16
"""
import torch
import json
import pandas as pd
from tqdm import tqdm
from config import set_args
from torch.utils.data.dataset import Dataset
from transformers.models.bert import BertTokenizer


args = set_args()


class SentencePairDataset(Dataset):
    def __init__(self, file_dir, is_train=True):
        # file_dir 多个数据的路径列表
        self.is_train = is_train
        self.aug_data = args.aug_data
        self.clip = args.clip_method
        self.len_limit = args.len_limit
        self.total_source_input_ids = []
        self.total_target_input_ids = []
        self.sample_types = []

        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

        lines = []
        for file in file_dir:
            with open(file, 'r', encoding='utf8') as f:
                for item in f.readlines():
                    line = json.loads(item.strip())
                    if 'labelA' in line:
                        line['type'] = 0
                        line['label'] = line['labelA']
                        del line['labelA']
                    else:
                        line['type'] = 1
                        line['label'] = line['labelB']
                        del line['labelB']
                    lines.append(line)

        content = pd.DataFrame(lines)
        content.columns = ['source', 'target', 'type', 'label']

        sources = content['source'].values.tolist()
        targets = content['target'].values.tolist()

        self.sample_types = content['type'].values.tolist()

        if self.is_train:
            self.labels = content['label'].values.tolist()
        else:
            self.ids = content['label'].values.tolist()

        for source, target in tqdm(zip(sources, targets), total=len(sources)):
            source = tokenizer.encode(source)[1:-1]   # 去掉[CLS] 和 [SEP]
            target = tokenizer.encode(target)[1:-1]

            if self.clip == 'head':
                if len(source) + 2 > self.len_limit:
                    source = source[0: self.len_limit - 2]
                if len(source) + 2 > self.len_limit:
                    target = target[0: self.len_limit - 2]
            if self.clip == 'tail':
                if len(source) + 2 > self.len_limit:
                    source = source[-self.len_limit + 2:]
                if len(target) + 2 > self.len_limit:
                    target = target[-self.len_limit + 2:]

            # 检查序列有没有超过限制
            assert len(source)+2 <= self.len_limit and len(target) + 2 <= self.len_limit

            # [CLS]:101, [SEP]:102
            source_input_ids = [101] + source + [102]
            target_input_ids = [101] + target + [102]

            assert len(source_input_ids) <= self.len_limit and len(target_input_ids) <= self.len_limit

            self.total_source_input_ids.append(source_input_ids)
            self.total_target_input_ids.append(target_input_ids)

        self.max_source_input_len = max([len(s) for s in self.total_source_input_ids])
        self.max_target_input_len = max([len(s) for s in self.total_target_input_ids])
        print("max source length: ", self.max_source_input_len)   # 第一个序列的最大长度
        print("max target length: ", self.max_target_input_len)   # 第二个序列的最大长度

    def __len__(self):
        return len(self.total_target_input_ids)

    def __getitem__(self, idx):
        source_input_ids = pad_to_maxlen(self.total_source_input_ids[idx], self.max_source_input_len)
        target_input_ids = pad_to_maxlen(self.total_target_input_ids[idx], self.max_target_input_len)
        sample_type = int(self.sample_types[idx])

        if self.is_train:
            label = int(self.labels[idx])
            return torch.LongTensor(source_input_ids), torch.LongTensor(target_input_ids), torch.LongTensor([label]), sample_type

        else:
            index = self.ids[idx]
            return torch.LongTensor(source_input_ids), torch.LongTensor(target_input_ids), index, sample_type


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    # 将序列padding到最大长度
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len-len(input_ids))
    return input_ids