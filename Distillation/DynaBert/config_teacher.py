# -*- encoding: utf-8 -*-
'''
@File    :   config_teacher.py
@Time    :   2020/10/19 09:53:10
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_model", default='./bert_pretrain/pytorch_model.bin', type=str, help='pretrain_model')
    parser.add_argument("--model_config", default='./bert_pretrain/bert_config.json', type=str, help='pretrain_model_config')
    parser.add_argument("--vocab_file", default='./bert_pretrain/vocab.txt')

    parser.add_argument("--train_data_path", default='./data/train_features.pkl.gz', type=str, help='data with training')
    parser.add_argument("--dev_data_path", default='./data/dev_features.pkl.gz', type=str, help='data with dev')


    parser.add_argument('--train_batch_size', type=int, default=32, help="random seed for initialization")
    parser.add_argument('--eval_batch_size', type=int, default=32, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=20, help="random seed for initialization")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="random seed for initialization")

    parser.add_argument('--n_gpu', type=int, default=1, help="random seed for initialization")


    parser.add_argument('--label_num', type=int, default=30, help="label is nums")
    parser.add_argument('--seed', type=int, default=43, help="random seed for initialization")

    # 保存老师模型
    parser.add_argument("--save_teacher_model", default='./save_teacher_model', type=str)
    args = parser.parse_args()
    return args
