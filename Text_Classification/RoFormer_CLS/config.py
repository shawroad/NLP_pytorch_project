"""
@file   : general_config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-22
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser(description='multi_label_cls model')
    parser.add_argument('--train_data', default='./data/train.csv', type=str, help='训练数据集')
    parser.add_argument('--train_batch_size', default=2, type=int, help='训练批次大小')
    parser.add_argument('--eval_batch_size', default=2, type=int, help='验证批次大小')

    parser.add_argument('--bert_vocab_path', default='./bert_pretrain/vocab.txt', type=str, help='bert词表的位置')
    parser.add_argument('--bert_pretrain_model', default='./bert_pretrain', type=str, help='预训练模型的路径')
    parser.add_argument('--learning_rate', default=1e-05, type=float, help='学习率大小')
    parser.add_argument('--num_epochs', default=10, type=int, help='训练几轮')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度累计')

    parser.add_argument('--outputs', default='./outputs', type=str, help='输出文件夹')
    return parser.parse_args()