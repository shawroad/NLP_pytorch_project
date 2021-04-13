"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-04-13
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", default='./data/train.csv', type=str)
    parser.add_argument("--test_data_path", default='./data/test.csv', type=str)
    parser.add_argument("--learning_rate", default=0.005, type=float,)
    parser.add_argument('--num_train_epochs', default=100, type=int)
    parser.add_argument('--use_pre_embed', default=True, type=bool)
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()
    return args
