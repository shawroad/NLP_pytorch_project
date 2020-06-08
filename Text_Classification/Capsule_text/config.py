"""

@file  : config.py

@author: xiaolu

@time  : 2020-06-04

"""
import torch


class Config:
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    vocab_size = 4502
    Epoch = 6
    batch_size = 32


