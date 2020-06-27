# -*- coding: utf-8 -*-
# @Time    : 2020/6/27 9:30
# @Author  : xiaolu
# @FileName: DataLoader.py
# @Software: PyCharm


import torch
import time
from datetime import timedelta
import json
from config import Config


def x_tokenize(ids):
    return [int(i) for i in ids]


def build_dataset(Config):
    '''
    加载数据
    :param config:
    :return:
    '''
    def load_dataset(path):
        contents = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                source = json.loads(line.strip())

                # 对两个东西进行padding
                token_ids = x_tokenize(source['input_ids'])

                mask = x_tokenize(source['input_mask'])
                max_len = 512
                if len(token_ids) < max_len:
                    token_ids.extend([0] * (max_len - len(token_ids)))

                if len(mask) < max_len:
                    mask.extend([0] * (max_len - len(mask)))
                ids = int(source['ids'])
                start = int(source['start_position'])
                end = int(source['end_position'])
                contents.append((ids, token_ids, mask, start, end))
        return contents

    train = load_dataset(Config.train_data_path)
    dev = load_dataset(Config.dev_data_path)
    return train, dev


class DatasetIterater:
    def __init__(self, data, Config):
        self.batch_size = Config.batch_size
        self.data = data
        self.n_batches = len(data) // self.batch_size
        self.residue = False
        if len(self.data) % self.n_batches != 0:
            self.residue = True

        self.index = 0
        self.device = Config.device

    def _to_tensor(self, datas):
        id = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        input_data = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        start = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        end = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        return id, input_data, mask, start, end

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.data[self.index * self.batch_size: len(self.data)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration

        else:
            batches = self.data[self.index * self.batch_size: (self.index + 1) * self.batch_size]

            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    build_dataset(Config)
