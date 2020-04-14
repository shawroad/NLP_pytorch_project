"""

@file  : config.py

@author: xiaolu

@time  : 2020-04-09

"""
import torch


class Config:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
    model_bert_path = './albert_pretrain/albert_large.bin'  # 训练好的模型　配置文件
    config_bert_path = './albert_pretrain/albert_config.json'
    learning_rate = 5e-5  # 学习率
    num_epochs = 100    # epoch数

    batch_size = 1

    seed = 42

    train_data_path = './data/train.data'
    dev_data_path = './data/dev.data'

