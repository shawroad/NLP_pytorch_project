# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2020/11/05 09:20:53
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import argparse


def set_args():
     parser = argparse.ArgumentParser()
     parser.add_argument("--is_word", default=False, type=bool, help='whether use word')
     parser.add_argument("--vocab_path", default="./data/vocab.pkl", type=str)
     parser.add_argument("--pad_size", default=512, type=int)
     parser.add_argument("--n_gram_vocab", default=250499, type=int)
     parser.add_argument("--n_vocab", default=6713, type=int)

     parser.add_argument("--train_batch_size", default=128, type=int)
     parser.add_argument("--eval_batch_size", default=128, type=int)

     parser.add_argument("--do_train", default=True, type=bool, help='whether use word')
     parser.add_argument("--num_train_epochs", default=20, type=int)
     parser.add_argument("--seed", default=43, type=int)
     parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="random seed for initialization")

     parser.add_argument("--train_features_path", default="./data/train_features.pkl.gz", type=str)
     parser.add_argument("--eval_features_path", default="./data/dev_features.pkl.gz", type=str)

     parser.add_argument("--save_teacher_model", default="./save_teacher_model", type=str)
     parser.add_argument("--save_log_file", default="./result_eval.txt", type=str)

     parser.add_argument("--embed_dim", default=300, type=int)
     parser.add_argument("--dropout", default=0.5, type=float)

     # hidden_size
     parser.add_argument("--hidden_size", default=256, type=int)

     parser.add_argument("--learning_rate", default=1e-3, type=float)
     # parser.add_argument("--fp16", default='O1', type=str)
     parser.add_argument("--fp16", default=False, type=bool)
     
     args = parser.parse_args()
     return args

     