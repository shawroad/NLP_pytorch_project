"""
# -*- coding: utf-8 -*-
# @File    : config.py
# @Time    : 2020/12/8 11:52 上午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features_path", default="./data/train_features.pkl.gz", type=str)
    parser.add_argument("--dev_features_path", default="./data/test_features.pkl.gz", type=str)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--is_train_embedding', type=bool, default=True)
    # 词嵌入维度和位置嵌入维度
    parser.add_argument('--embed_dim', type=int, default=100, help='词嵌入的维度')
    parser.add_argument('--pos_dim', type=int, default=25, help='位置嵌入的维度')
    parser.add_argument('--vocab_size', type=int, default=3123, help='词表的大小')
    parser.add_argument('--pos_size', type=int, default=102, help='位置的最大值')
    parser.add_argument('--tag_size', type=int, default=12, help='关系的个数')

    parser.add_argument('--hidden_size', type=int, default=200, help='lstm hidden dim')
    parser.add_argument("--save_model", default="./save_model", type=str)
    args = parser.parse_args()
    return args



