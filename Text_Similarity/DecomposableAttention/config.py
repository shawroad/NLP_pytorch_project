"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-07
"""
import argparse


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=32, help='训练批次大小')
    parser.add_argument('--dev_batch_size', type=int, default=32, help='验证批次大小')

    parser.add_argument('--max_char_len', default=40, type=int, help='每个句子的最大长度')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='模型训练时的学习率')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='总共训练几轮')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='梯度裁剪的范围')
    parser.add_argument('--save_model', default='output', type=str, help='模型保存的位置')
    return parser.parse_args()
