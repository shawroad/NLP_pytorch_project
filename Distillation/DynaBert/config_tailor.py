# -*- encoding: utf-8 -*-
'''
@File    :   config_tailor.py
@Time    :   2020/10/19 10:22:04
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import argparse


def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path", default='./data/train_features.pkl.gz', type=str, help='data with training')
    parser.add_argument("--dev_data_path", default='./data/dev_features.pkl.gz', type=str, help='data with dev')

    # 宽度瘦身的力度
    parser.add_argument('--width_mult_list', type=str, default='0.25,0.5,0.75,1.0',
                        help="the possible widths used for training, e.g., '1.' is for separate training "
                             "while '0.25,0.5,0.75,1.0' is for vanilla slimmable training")
    # 高度减小的力度
    parser.add_argument('--depth_mult_list', type=str, default='0.5,0.75,1.0',
                        help="the possible depths used for training, e.g., '1.' is for default")
    
    parser.add_argument('--n_gpu', type=int, default=1, help="random seed for initialization")

    parser.add_argument("--pretrain_model", default='./bert_pretrain/pytorch_model.bin', type=str, help='pretrain_model')
    parser.add_argument("--model_config", default='./bert_pretrain/bert_config.json', type=str, help='pretrain_model_config')
    parser.add_argument("--vocab_file", default='./bert_pretrain/vocab.txt')

    parser.add_argument("--do_lower_case", default=True,help="Set this flag if you are using an uncased model.")
    parser.add_argument('--train_batch_size', type=int, default=8, help="random seed for initialization")
    parser.add_argument('--eval_batch_size', type=int, default=8, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="random seed for initialization")
    parser.add_argument('--logging_steps', type=int, default=2000, help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=20, help="random seed for initialization")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="random seed for initialization")

    parser.add_argument("--save_student_model", default='./save_student_model', type=str, help='pretrain_model')
    parser.add_argument("--save_teacher_model", default='./save_teacher_model', type=str, help='pretrain_model')

    parser.add_argument('--seed', type=int, default=43, help="random seed for initialization")

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")


    args = parser.parse_args()
    return args
