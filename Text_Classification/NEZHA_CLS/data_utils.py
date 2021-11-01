"""
@file   : data_utils.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-09-30
"""
import os
import torch
import json
import random
from itertools import chain
from config import set_args
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers.models.bert import BertTokenizer
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter


class TxtDataSet(Dataset):
    def __init__(self, data_set_name='train', path=''):
        self.args = set_args()
        self.tokenizer = BertTokenizer.from_pretrained('./nezha_pretrain')
        self.label2id = {'angry': 0, 'happy': 1, 'neutral': 2, 'surprise': 3, 'sad': 4, 'fear': 5}
        self.id2label = {0: "angry", 1: "happy", 2: "neutral", 3: "surprise", 4: "sad", 5: "fear"}

        save_features_file = os.path.join(self.args.data_dir, 'features_{}_max_len_{}'.format(data_set_name, self.args.max_len))
        if os.path.exists(save_features_file):
            print('加载已处理过的数据')
            self.data_set = torch.load(save_features_file)['data_set']
        else:
            print('对数据进行预处理')
            self.data_set = self.load_data(path)
            torch.save({'data_set': self.data_set}, save_features_file)

    def load_data(self, path):
        data_set = []
        with open(path, 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                sample = json.loads(line.strip())
                input_ids, attention_mask, label = self.convert_featrue(sample)
                sample['input_ids'] = input_ids
                sample['attention_mask'] = attention_mask
                sample['label'] = label
                data_set.append(sample)
        return data_set

    def convert_featrue(self, sample):
        label = self.label2id[sample["label"]]
        tokens = self.tokenizer.tokenize(sample["text"])
        if len(tokens) > self.args.max_len - 2:
            tokens = tokens[:self.args.max_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(attention_mask)
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


class BlockShuffleDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, sort_key, sort_bs_num=None, is_shuffle=True, **kwargs):
        """
        初始化函数，继承DataLoader类
        Args:
            dataset: Dataset类的实例，其中中必须包含dataset变量，并且该变量为一个list
            sort_key: 排序函数，即使用dataset元素中哪一个变量的长度进行排序
            sort_bs_num: 排序范围，即在多少个batch_size大小内进行排序，默认为None，表示对整个序列排序
            is_shuffle: 是否对分块后的内容，进行随机打乱，默认为True
            **kwargs:
        """
        super(BlockShuffleDataLoader, self).__init__(dataset, **kwargs)
        self.sort_bs_num = sort_bs_num
        self.sort_key = sort_key
        self.is_shuffle = is_shuffle

    def __iter__(self):
        self.dataset.data_set = self.block_shuffle(self.dataset.data_set, self.batch_size, self.sort_bs_num,
                                                   self.sort_key, self.is_shuffle)
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @staticmethod
    def block_shuffle(data, batch_size, sort_bs_num, sort_key, is_shuffle):
        random.shuffle(data)
        # 将数据按照batch_size大小进行切分
        tail_data = [] if len(data) % batch_size == 0 else data[-len(data) % batch_size:]
        data = data[:len(data) - len(tail_data)]
        assert len(data) % batch_size == 0
        # 获取真实排序范围
        sort_bs_num = len(data) // batch_size if sort_bs_num is None else sort_bs_num
        # 按照排序范围进行数据划分
        data = [data[i:i + sort_bs_num * batch_size] for i in range(0, len(data), sort_bs_num * batch_size)]
        # 在排序范围，根据排序函数进行降序排列
        data = [sorted(i, key=sort_key, reverse=True) for i in data]
        # 将数据根据batch_size获取batch_data
        data = list(chain(*data))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        # 判断是否需要对batch_data序列进行打乱
        if is_shuffle:
            random.shuffle(data)
        # 将tail_data填补回去
        data = list(chain(*data)) + tail_data
        return data


def collate_fn(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, attention_mask_list, labels_list = [], [], []
    for instance in batch_data:
        input_ids_temp = instance["input_ids"]
        attention_mask_temp = instance["attention_mask"]
        labels_temp = instance["label"]
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
        labels_list.append(labels_temp)
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "label": torch.tensor(labels_list, dtype=torch.long)}
