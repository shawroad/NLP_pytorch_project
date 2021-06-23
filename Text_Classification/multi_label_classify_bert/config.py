"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-22
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='./data/train.csv', type=str, help='训练数据集')
    parser.add_argument('--test_data', default='./data/test.csv', type=str, help='测试数据集')
    parser.add_argument('--num_classes', default=25551, type=int, help='标签数')
    parser.add_argument('--num_epoch', default=50, type=int, help='训练几轮')

    parser.add_argument('--train_batch_size', default=2, type=int, help='训练批次的大小')
    parser.add_argument('--test_batch_size', default=2, type=int, help='测试批次的大小')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='学习率大小')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减')

    # 模型
    parser.add_argument('--vocab', default='./bert_pretrain/vocab.txt', type=str, help='bert词表')
    parser.add_argument('--bert_pretrain', default='./bert_pretrain/pytorch_model.bin', type=str, help='bert权重')
    parser.add_argument('--bert_config', default='./bert_pretrain/bert_config.json', type=str, help='bert的配置文件')


    return parser.parse_args()
