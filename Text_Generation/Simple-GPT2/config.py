"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-11
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_num', type=int, default=200, help='输入内容最大长度')
    parser.add_argument('--stride', type=int, default=5, help='偏移量大小')
    parser.add_argument('--multi', type=int, default=4, help='多头注意力中的头的个数')
    parser.add_argument('--head_num', type=int, default=12, help='每个头的维度')
    parser.add_argument('--embed_dim', type=int, default=60, help='词嵌入的维度')
    parser.add_argument('--vocab_num', type=int, default=1032, help='词表的大小')  # 根据语料 构建的词表大小不同
    parser.add_argument('--type_num', type=int, default=15, help='句编码，可以用15个句子')
    parser.add_argument('--block_num', type=int, default=6, help='定义解码器的数量')

    parser.add_argument('--num_train_epoch', type=int, default=30, help='定义解码器的数量')
    parser.add_argument('--save_path', type=str, default='output', help='模型保存位置')
    return parser.parse_args()

