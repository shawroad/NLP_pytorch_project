"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-30$
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-max_seq_length', default=512, type=int)
    parser.add_argument('-batch_size', default=12, type=int)
    parser.add_argument('-num_epochs', default=4, type=int)
    parser.add_argument('-learning_rate', default=2e-5, type=float)
    parser.add_argument('-max_grad_norm', default=1.0, type=float)
    parser.add_argument('-warm_up_proportion', default=0.1, type=float)
    parser.add_argument('-gradient_accumulation_step', default=1, type=int)
    parser.add_argument('-bert_path', default='../bert_pretrain')
    parser.add_argument('-trunc_mode', default=128, type=str, help='可选head tail 以及某个数字k(取前k个+以及从后面取余下的部分)')
    args = parser.parse_args()
    return args
