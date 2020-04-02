"""

@file  : config.py

@author: xiaolu

@time  : 2020-04-01

"""
import torch


class Config:
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    MAX_LENGTH = 10  # Maximum sentence length to consider
    MIN_COUNT = 3  # Minimum word count threshold for trimming

    # Configure models
    model_name = 'cb_model'
    attn_model = 'dot'
    # attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 5

    total_step = 100000

    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0



