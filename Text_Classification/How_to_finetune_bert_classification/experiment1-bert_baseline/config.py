"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-29$
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-max_seq_length', default=512, type=int)
    parser.add_argument('-bert_path', default='../bert_pretrain', type=str)
    parser.add_argument('-learning_rate', default=2e-5, type=float)
    parser.add_argument('-batch_size', default=24, type=int)
    parser.add_argument('-num_epochs', default=4, type=int)
    parser.add_argument('-max_grad_norm', default=1.0, type=float)
    parser.add_argument('-warm_up_proportion', default=0.1, type=float)
    parser.add_argument('-gradient_accumulation_step', default=1, type=int)
    args = parser.parse_args()
    return args
