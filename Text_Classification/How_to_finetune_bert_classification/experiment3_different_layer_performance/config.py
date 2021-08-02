"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-30$
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', default=0, type=int, help='随机种子的取值')
    parser.add_argument('-max_seq_length', default=512, type=int, help='输入的最大长度')
    parser.add_argument('-batch_size', default=12, type=int, help='训练批次的大小')
    parser.add_argument('-num_epochs', default=4, type=int, help='总共将数据训练几遍')
    parser.add_argument('-learning_rate', default=2e-5, type=float, help='学习率大小')
    parser.add_argument('-max_grad_norm', default=1.0, type=float, help='梯度裁剪的大小')
    parser.add_argument('-warm_up_proportion', default=0.1, type=float, help='预热步数占总步数的比例')
    parser.add_argument('-gradient_accumulation_step', default=1, type=int, help='梯度积聚的大小')
    parser.add_argument('-bert_path', default='../bert_pretrain', help='预训练模型的路径')
    parser.add_argument('-trunc_mode', default=128, type=str, help='从开始取128个，然后剩余的在最后取')
    parser.add_argument('-num_pool_layers', default=4, type=int, help='使用最后几层进行池化')
    parser.add_argument('-pool_mode', default="concat", type=str, help='最后几层怎样处理 concat or mean or max')
    args = parser.parse_args()
    return args

