# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 18:07
# @Author  : xiaolu
# @FileName: config.py
# @Software: PyCharm
import torch

class Config:
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    max_ngram = 3
    data = './corpus/corpus.txt'
    output_dir = './corpus/'

    config_path = './bert_pretrain/bert_config.json'
    adam_epsilon = 1e-8
    learning_rate = 0.000176

    epochs = 10
    warmup_proportion = 0.1
    gradient_accumulation_steps = 1
    max_grad_norm = 1
    num_eval_steps = 2   # 多少步进行数据的预测
    num_save_steps = 50  # 多少步进行模型的保存


