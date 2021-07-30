# -*- coding: utf-8 -*-
# @Time    : 2020/8/26 17:25
# @Author  : xiaolu
# @FileName: config.py
# @Software: PyCharm
import torch


class Config:
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    num_train_epochs = 10

    train_batch_size = 2
    eval_batch_size = 3


    gradient_accumulation_steps = 1
    learning_rate = 1e-5

    ckpt_dir = './save_model'
