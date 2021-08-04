"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-04
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./bert_pretrain', type=str)
    parser.add_argument('--train_data_path', default='./data/train_proceed.txt')
    parser.add_argument('--dev_data_path', default='./data/dev_proceed')
    parser.add_argument('--test_data_path', default='./data/test_proceed')
    parser.add_argument('--output_dir',  default="./outputs", type=str)
    parser.add_argument('--batch_size',  default=3, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--maxlen', default=40, type=int)
    parser.add_argument('--early_stop', default=5, type=int)
    parser.add_argument('--num_train_epochs', default=3, type=int)
    args = parser.parse_args()
    return args

