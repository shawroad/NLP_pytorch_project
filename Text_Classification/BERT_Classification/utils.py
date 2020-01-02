"""

@file   : utils.py

@author : xiaolu

@time   : 2019-12-31

"""
import torch
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    '''
    加载数据
    :param config:
    :return:
    '''
    def load_dataset(path, pad_size=32):
        '''
        :param path: 数据路径
        :param pad_size: 想把文本padding成的尺寸
        :return:
        '''
        contents = []
        with open(path, 'r', encoding='utf8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, label = line.split('\t')   # 分出文本与标签
                token = config.tokenizer.tokenize(content)  # 转id
                token = [CLS] + token

                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
                # 返回内容包括: 文本转为id序列, 标签, 当前文本的真实长度, mask向量(padding部位填充为0 其余位置填充为1)
        return contents
    dev = load_dataset(config.dev_path, config.pad_size)
    train = load_dataset(config.train_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


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


def build_iterator(dataset, config):
    '''
    数据迭代器
    :param dataset:
    :param config:
    :return:
    '''
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))