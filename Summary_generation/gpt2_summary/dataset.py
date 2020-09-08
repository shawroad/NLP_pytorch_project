# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 13:56
# @Author  : xiaolu
# @FileName: dataset.py
# @Software: PyCharm

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, item):
        input_ids = self.data_list[item].strip()
        input_ids = [int(token_id) for token_id in input_ids.split()]
        return input_ids

    def __len__(self):
        return len(self.data_list)

