"""

@file  : config.py

@author: xiaolu

@time  : 2020-03-04

"""
import torch


class Config:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
    model_bert_path = './bert-base-chinese.tar.gz'  # 训练好的模型　配置文件
    learning_rate = 5e-5  # 学习率
    num_epochs = 4    # epoch数

    batch_size = 4

    seed = 42

    train_data_path = './dataset/train.data'
    dev_data_path = './dataset/dev.data'



