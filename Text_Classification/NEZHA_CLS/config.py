"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-09-30
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--分类demo')
    parser.add_argument('--max_len', default=100, type=int, help='文本的最大的长度')
    parser.add_argument('--data_dir', default='./data', type=str, help='数据保存')
    parser.add_argument('--train_batch_size', default=8, type=int, help='批次大小')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='训练几轮')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积聚')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    return parser.parse_args()
