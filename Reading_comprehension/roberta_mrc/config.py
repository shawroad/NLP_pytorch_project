# -*- coding: utf-8 -*-
# @Time    : 2020/6/27 9:29
# @Author  : xiaolu
# @FileName: config.py
# @Software: PyCharm

import torch


class Config:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
    model_path = './roberta_pretrain/pytorch_model.bin'
    config_path = './roberta_pretrain/bert_config.json'

    learning_rate = 5e-5  # 学习率
    num_epochs = 100    # epoch数

    batch_size = 2
    seed = 42

    train_data_path = './data/train.data'
    dev_data_path = './data/dev.data'