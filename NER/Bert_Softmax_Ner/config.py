"""

@file  : config.py

@author: xiaolu

@time  : 2020-05-25

"""
import torch

class Config:
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    base_path = './data/msra'    # 基路径

    model_path = './bert_pretrain/pytorch_model.bin'
    model_vocab_path = './bert_pretrain/vocab.txt'
    model_config_path = './bert_pretrain/config.json'

    batch_size = 2
    max_len = 180
    learning_rate = 3e-5

    epoch_num = 20
    num_tag = 7
    clip_grad = 2


