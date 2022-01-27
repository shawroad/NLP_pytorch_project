"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-01-05
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--xxx分类')
    parser.add_argument('--train_data', default='./data/data.csv', type=str, help='训练数据集')
    parser.add_argument('--pretrained_model_path', default='./nezha_pretrain', type=str, help='预训练模型的路径')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='模型输出')

    parser.add_argument('--num_train_epochs', default=5, type=int, help='训练几轮')
    parser.add_argument('--train_batch_size', default=2, type=int, help='训练批次大小')
    parser.add_argument('--val_batch_size', default=4, type=int, help='验证批次大小')
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int, help='梯度积累几次更新')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='学习率大小')
    parser.add_argument('--seed', default=43, type=int, help='随机种子')

    return parser.parse_args()

