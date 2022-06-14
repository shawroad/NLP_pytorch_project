"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-14
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--预训练')
    parser.add_argument('--train_data_path', default='./data/text.txt', type=str, help='训练数据集')
    parser.add_argument('--pretrain_weight', default='./roberta_pretrain', type=str, help='预训练模型的路径')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='模型输出')

    parser.add_argument('--num_train_epochs', default=20, type=int, help='训练几轮')
    parser.add_argument('--weight_decay_rate', default=0.01, type=float, help='权重衰减的步数比例')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='优化器参数')
    parser.add_argument('--train_batch_size', default=2, type=int, help='训练批次大小')
    parser.add_argument('--val_batch_size', default=32, type=int, help='验证批次大小')
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int, help='梯度积累几次更新')
    parser.add_argument('--learning_rate', default=5e-6, type=float, help='学习率大小')
    parser.add_argument('--seed', default=43, type=int, help='随机种子')
    parser.add_argument('--save_checkponint_steps', default=2, type=int, help='保存多少步的checkpoint')
    return parser.parse_args()