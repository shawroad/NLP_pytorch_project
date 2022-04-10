"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-04-06
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--NER')
    parser.add_argument('--train_data_path', default='./data/train.json', type=str, help='训练数据')
    parser.add_argument('--valid_data_path', default='./data/dev.json', type=str, help='验证数据')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='初始化学习率')
    parser.add_argument('--output_dir', default='output', type=str, help='模型保存位置')
    parser.add_argument('--num_epochs', default=50, type=int, help='训练轮次')
    parser.add_argument('--batch_size', default=64, type=int, help='批次大小')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累几次更新')
    parser.add_argument('--max_seq_len', default=64, type=int, help='最大长度')
    return parser.parse_args()
