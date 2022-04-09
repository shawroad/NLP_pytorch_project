"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-31
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output', type=str, help='模型保存路径')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpt2_path', default='./gpt2_pretrain')
    parser.add_argument("--lr", type=float, default=3e-5, help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--dev_size', type=int, default=1000)
    parser.add_argument('--prefix_len', type=int, default=10)
    parser.add_argument('--clip_size', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--eval_step", type=int, default=10000, help="训练多少步,记录一次指标")
    args = parser.parse_args()
    return args
