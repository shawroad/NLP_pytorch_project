"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-06
"""
import argparse


def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=12, help='训练批次大小')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='总共训练几轮')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='模型训练时的学习率')
    parser.add_argument('--logging_steps', default=20, type=int, help='保存训练日志的步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--eval_steps', default=5, type=int, help='训练时，多少步进行一次测试')
    return parser.parse_args()
