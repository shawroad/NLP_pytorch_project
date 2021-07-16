"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-16
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/', type=str, help="数据保存路径")
    parser.add_argument("--save_dir", default='./outputs/', type=str, help="结果保存路径")
    parser.add_argument("--tokenizer_path", default='./model_weight/NEZHA/vocab.txt', type=str, help='词表的位置')

    parser.add_argument("--aug_data", default=True, type=bool, help="是否进行数据增强")
    parser.add_argument("--use_fgm", default=False, type=bool, help="是否使用fgm对抗训练")
    parser.add_argument("--clip_method", default='tail', type=str, help="序列阶段的方法 从头或者尾截断，可选head或tail")
    parser.add_argument("--len_limit", default=512, type=int, help="单个输入序列的最大长度")

    parser.add_argument("--epochs", default=10, type=int, help='将数据训练几轮')
    parser.add_argument("--learning_rate", default=2e-5, type=float, help='学习率的大小')
    parser.add_argument("--weight_decay", default=1e-3, type=float, help='')

    # hidden_size = config.hidden_size
    parser.add_argument('--train_batch_size', default=4, type=int, help='训练的批次大小')
    parser.add_argument('--eval_batch_size', default=4, type=int, help='验证的批次大小')

    parser.add_argument('--task_type', default='ab', type=str, help='是否将两种数据分开，可设置为a, b, ab')

    parser.add_argument('--use_scheduler', default=True, type=bool, help='学习率动态变化 要不要呢？')




    # task_type
    return parser.parse_args()


