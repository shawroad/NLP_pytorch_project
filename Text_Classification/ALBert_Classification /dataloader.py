"""

@file  : dataloader.py

@author: xiaolu

@time  : 2020-03-19

"""
import torch


class DatasetIterater:
    def __init__(self, batches, batch_size, device):
        '''
        :param batches: datasets 数据集
        :param batch_size: 批量的大小
        :param device: 指定设备
        '''
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size   # 按当前batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)  # 文本转为的id序列
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)  # 标签

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)  # 真实文本的长度
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)  # mask向量
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
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
