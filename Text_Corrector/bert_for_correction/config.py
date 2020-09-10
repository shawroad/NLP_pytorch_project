# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 18:07
# @Author  : xiaolu
# @FileName: config.py
# @Software: PyCharm
import torch

class Config:
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    max_ngram = 3
    data = './data/train.data'
    output_dir = './data/'

    config_path = './bert_base_pretrain/config.json'
    adam_epsilon = 1e-8
    learning_rate = 0.000176

    epochs = 50
    warmup_proportion = 0.1
    gradient_accumulation_steps = 1
    max_grad_norm = 1
    num_eval_steps = 10000   # 多少步进行数据的预测
    num_save_steps = 10000  # 多少步进行模型的保存


