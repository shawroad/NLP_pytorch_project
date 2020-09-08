"""

@file  : config.py

@author: xiaolu

@time  : 2020-04-14

"""
import torch


class Config:
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    vocab_size = 2672
    max_len = 492
    max_ans_len = 62

    PAD = 0  # Used for padding short sentences
    SOS = 1  # Start-of-sentence token
    EOS = 2  # End-of-sentence token
    UNK = 3

    batch_size = 2
    num_epochs = 100

    ans_max_len = 30

    hidden_size = 512
    encoder_n_layers = 2
    decoder_n_layers = 1
    dropout = 0.1
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0


    train_data_path = './data/train.json'


