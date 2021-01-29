"""
# -*- coding: utf-8 -*-
# @File    : config.py
# @Time    : 2021/1/26 1:58 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", default="./roberta_pretrain/vocab.txt", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--train_features_path", default="./data/mini_data/dev_features.pkl.gz", type=str)
    parser.add_argument("--eval_features_path", default="./data/mini_data/dev_features.pkl.gz", type=str)

    parser.add_argument('--train_batch_size', type=int, default=2, help="random seed for initialization")
    parser.add_argument('--eval_batch_size', type=int, default=2, help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="random seed for initialization")
    parser.add_argument('--num_train_epochs', default=20, type=str, help='code will operate in this gpu')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="random seed for initialization")

    # ckpt_dir
    parser.add_argument("--save_teacher_model", default='./save_teacher_model', type=str)
    # parser.add_argument("--save_student_model", default="./save_student_model", type=str)

    parser.add_argument('--alpha', type=float, default=0.3, help="random seed for initialization")
    args = parser.parse_args()
    return args
