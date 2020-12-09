"""
# -*- coding: utf-8 -*-
# @File    : config.py
# @Time    : 2020/12/8 5:25 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--char_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=20)

    # parser.add_argument("--dev_features_path", default="./data/test_features.pkl.gz", type=str)
    # parser.add_argument('--learning_rate', type=float, default=0.005)
    # parser.add_argument('--num_train_epochs', type=int, default=20)

    args = parser.parse_args()
    return args



