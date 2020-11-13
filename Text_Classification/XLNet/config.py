# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2020/10/30 19:56:11
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import argparse


def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_features_path", default="./data/train_features.pkl.gz", type=str)
    parser.add_argument("--eval_features_path", default="./data/dev_features.pkl.gz", type=str)

    parser.add_argument("--save_teacher_model", default="./save_teacher_model", type=str)
    parser.add_argument("--save_log_file", default="./result_eval.txt", type=str)

    # num_train_epochs
    parser.add_argument('--num_train_epochs', default=20, type=str, help='code will operate in this gpu')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--vocab_file", default="./xlnet_pretrain/spiece.model", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument('--train_batch_size', type=int, default=2, help="random seed for initialization")
    parser.add_argument('--eval_batch_size', type=int, default=2, help="random seed for initialization")

    parser.add_argument('--n_gpu', type=int, default=1, help="random seed for initialization")

    # gradient_accumulation_steps
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="random seed for initialization")
    # do_train
    parser.add_argument('--do_train', type=bool, default=True, help="random seed for initialization")
    
    # learning_rate
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="random seed for initialization")
    parser.add_argument('--alpha', type=float, default=0.3, help="random seed for initialization")
    parser.add_argument('--fp16', type=str, default="O1")
    # parser.add_argument('--fp16', type=bool, default=False)
    args = parser.parse_args()
    return args
