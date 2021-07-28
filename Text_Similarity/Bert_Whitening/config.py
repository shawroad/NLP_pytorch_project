"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-28$
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../WeiBo_CLS/bert_pretrain', type=str, required=False)
    parser.add_argument('--save_path', type=str, default="./output/whitening")
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--pooling', type=str, default="first_last")
    parser.add_argument('--dim', type=int, default=768)
    args = parser.parse_args()

    return args