# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 19:35
# @Author  : xiaolu
# @FileName: train.py
# @Software: PyCharm

import pickle
import pdb
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model import NERLSTM_CRF
from utils import get_tags, format_result
from config import Config
from rlog import _log_normal, _log_warning, _log_info, _log_error, _log_toomuch, _log_bg_blue, _log_bg_pp, _log_fg_yl, \
    _log_fg_cy, _log_black, rainbow
import time


class NERDataset(Dataset):
    def __init__(self, X, Y):
        self.data = [{'x': X[i], 'y': Y[i]} for i in range(X.shape[0])]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def train():
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)

    for epoch in range(Config.max_epoch):
        model.train()
        model.to(Config.device)
        for index, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            X = batch['x'].long().to(Config.device)    # torch.Size([4, 60])    (batch_size, max_len)
            y = batch['y'].long().to(Config.device)    # torch.Size([4, 60])    (batch_size, max_len)

            # CRF
            loss = model.log_likelihood(X, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
            optimizer.step()

            now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            o_str = 'time: {}, epoch: {}, step: {}, loss: {:6f}'.format(now_time, epoch, index, loss.item())
            rainbow(o_str)

        aver_loss = 0
        preds, labels = [], []
        for index, batch in enumerate(valid_dataloader):
            model.eval()
            val_x, val_y = batch['x'].long().to(Config.device), batch['y'].long().to(Config.device)
            predict = model(val_x)
            # CRF
            loss = model.log_likelihood(val_x, val_y)
            aver_loss += loss.item()
            # 统计非0的，也就是真实标签的长度
            leng = []
            for i in val_y.cpu():
                tmp = []
                for j in i:
                    if j.item() > 0:
                        tmp.append(j.item())
                leng.append(tmp)

            for index, i in enumerate(predict):
                preds += i[:len(leng[index])]

            for index, i in enumerate(val_y.tolist()):
                labels += i[:len(leng[index])]
        aver_loss /= (len(valid_dataloader) * 64)
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')
        report = classification_report(labels, preds)
        print(report)
        torch.save(model.state_dict(), './save_model/bilstm_ner.bin')


def predict(tag, input_str=""):
    model.load_state_dict(torch.load('./save_model/bilstm_ner.bin'))
    if not input_str:
        input_str = input("请输入文本: ")
    input_vec = [word2id.get(i, 0) for i in input_str]

    # convert to tensor
    sentences = torch.tensor(input_vec).view(1, -1)
    paths = model(sentences)

    entities = []
    tags = get_tags(paths[0], tag, tag2id)
    entities += format_result(tags, input_str, tag)
    print(entities)


if __name__ == '__main__':
    # 1. 加载数据集
    with open(Config.pickle_path, 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)
    print("train len:", len(x_train))   # train len: 24271
    print("test len:", len(x_test))    # test len: 7585
    print("valid len", len(x_valid))   # valid len 6068

    train_dataset = NERDataset(x_train, y_train)
    valid_dataset = NERDataset(x_valid, y_valid)
    test_dataset = NERDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)

    model = NERLSTM_CRF(Config.embedding_dim, Config.hidden_dim, Config.dropout, word2id, tag2id)
    # train()   # 训练

    predict(tag='ns', input_str='我从西安来，你爷爷的大名叫路路, 来给我预测预测。')
