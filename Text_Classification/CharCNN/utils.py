"""

@file  : utils.py

@author: xiaolu

@time  : 2019-10-28

"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


class MyDataset(Dataset):
    # 继承Dataset类　必须实现__getitem__()和　__len__()
    def __init__(self, data_path, config):
        self.config = config
        self.vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        self.identity_mat = np.identity(len(self.vocabulary))  # 对角阵 类似于one_hot编码
        data = get_pandas_df(data_path)
        self.texts = list(data.text)
        self.labels = list(data.label)
        self.length = len(self.labels)

    def __getitem__(self, index):
        # 1. 取出文本
        raw_text = self.texts[index]

        # 2. 将文本转为一个one_hot 矩阵
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.float32)

        if len(data) > self.config.max_len:
            # 若大于我们规定的最大长度 直接截断
            data = data[:self.config.max_len]

        elif 0 < len(data) < self.config.max_len:
            # 长度不够填充零
            data = np.concatenate(
                (data,
                 np.zeros((self.config.max_len - len(data), len(self.vocabulary)), dtype=np.float32)
            ))
        elif len(data) == 0:
            data = np.zeros((self.config.max_len, len(self.vocabulary)), dtype=np.float32)

        label = self.labels[index]
        return data, label

    def __len__(self):
        return self.length


def parse_label(label):
    '''
    Input:
        label (string) : labels of the form '__label__2'
    Returns:
        label (int) : integer value corresponding to label string
    '''
    return int(label.strip()[-1]) - 1


def get_pandas_df(filename):
    '''
    加载数据
    :param filename:
    :return:
    '''
    with open(filename, 'r') as datafile:
        data = [line.strip().split(',', maxsplit=1) for line in datafile]
        data_text = list(map(lambda x: x[1], data))
        data_label = list(map(lambda x: parse_label(x[0]), data))

    full_df = pd.DataFrame({'text': data_text, 'label': data_label})
    return full_df


def get_iterators(config, train_file, test_file, val_file=None):
    train_set = MyDataset(train_file, config)
    test_set = MyDataset(test_file, config)

    # 若有验证集则用验证集  没有的话从训练集中截取一部分
    if val_file:
        val_set = MyDataset(val_file, config)
    else:
        train_size = int(0.9 * len(train_set))
        test_size = len(train_set) - train_size
        train_set, val_set = data.random_split(train_set, [train_size, test_size])  # 从训练集中随机切10%给验证集

    train_iterator = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_iterator = DataLoader(test_set, batch_size=config.batch_size)
    val_iterator = DataLoader(val_set, batch_size=config.batch_size)
    return train_iterator, test_iterator, val_iterator


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx, batch in enumerate(iterator):
        if torch.cuda.is_available():
            batch = [Variable(record).cuda() for record in batch]
        else:
            batch = [Variable(record) for record in batch]

        x, y = batch
        y_pred = model(x)
        predicted = torch.max(y_pred.data, 1)[1]
        all_preds.extend(predicted.numpy())
        all_y.extend(y.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score
