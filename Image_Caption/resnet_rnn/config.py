"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-25
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--Image2Caption')
    parser.add_argument('--epochs', default=200, type=float, help='训练和验证的批次大小')
    parser.add_argument('--max_len', default=40, type=int, help='文本描述的最大长度')
    parser.add_argument('--batch_size', default=64, type=int, help='训练和验证的批次大小')

    parser.add_argument('--alpha_c', default=1., type=float, help='损失权重')

    parser.add_argument('--decoder_lr', default=1e-4, type=float, help='解码器学习率')
    parser.add_argument('--encoder_lr', default=4e-4, type=float, help='编码器学习率')

    parser.add_argument('--grad_clip', default=5., type=float, help='梯度裁剪')
    parser.add_argument('--print_freq', default=1000, type=int, help='多少次打印一下')

    # 模型的超参数
    parser.add_argument('--attention_dim', default=512, type=int, help='训练和验证的批次大小')
    parser.add_argument('--emb_dim', default=512, type=int, help='训练和验证的批次大小')
    parser.add_argument('--decoder_dim', default=512, type=int, help='训练和验证的批次大小')
    parser.add_argument('--dropout', default=0.5, type=float, help='训练和验证的批次大小')

    args = parser.parse_args()
    return args

