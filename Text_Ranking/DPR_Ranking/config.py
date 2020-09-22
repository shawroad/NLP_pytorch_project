# -*- coding: utf-8 -*-
# @Time    : 2020/9/17 9:41
# @Author  : xiaolu
# @FileName: config.py
# @Software: PyCharm
import argparse


def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='5', type=str, help='code will operate in this gpu')
    # num_train_epochs
    parser.add_argument('--num_train_epochs', default=10, type=str, help='code will operate in this gpu')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--vocab_file", default="./roberta_pretrain/vocab.txt", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--output_dir", default="./output", type=str)
    # ckpt_dir
    parser.add_argument("--ckpt_dir", default="./save_model", type=str)

    parser.add_argument('--train_batch_size', type=int, default=3, help="random seed for initialization")
    parser.add_argument('--eval_batch_size', type=int, default=2, help="random seed for initialization")

    # gradient_accumulation_steps
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="random seed for initialization")
    parser.add_argument("--train_features_path", default="./data/train_features.pkl.gz", type=str)
    parser.add_argument("--eval_features_path", default="./data/dev_features.pkl.gz", type=str)
    # do_train
    parser.add_argument('--do_train', type=bool, default=True, help="random seed for initialization")
    # learning_rate
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="random seed for initialization")
    args = parser.parse_args()
    return args








