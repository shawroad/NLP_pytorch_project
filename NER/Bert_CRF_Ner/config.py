# -*- coding: utf-8 -*-
# @Time    : 2020/9/4 14:34
# @Author  : xiaolu
# @FileName: config.py
# @Software: PyCharm
import argparse


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="2", help="For distant debugging.")
    parser.add_argument("--data_dir", type=str, default='./data', help="this is dataset path")
    parser.add_argument("--output_dir", default='./outputs', type=str, required=False, help='this output directory')
    parser.add_argument("--checkpoint_path", type=str, default='./save_model', help="For distant debugging.")
    parser.add_argument("--pred_output_dir", type=str, default='./output_predict', help="For distant debugging.")

    # pred_output_dir

    parser.add_argument("--max_seq_length", type=int, default=128, help='the length of sequence')
    parser.add_argument("--do_train", type=bool, default=True, help="For distant debugging.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="For distant debugging.")
    parser.add_argument("--predict_batch_size", type=int, default=64, help="For distant debugging.")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="For distant debugging.")

    parser.add_argument("--epochs", type=int, default=10, help="For distant debugging.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="For distant debugging.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help='the initial learning rate for Adam')
    parser.add_argument("--crf_learning_rate", default=5e-5, type=float, help='the initial learning rate for '
                                                                              'crf and linear layer')

    parser.add_argument("--warmup_proportion", default=0.05, type=float, help='the initial learning rate for Adam')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help='Epsilon for Adam optimizer')
    parser.add_argument("--weight_decay", default=0.01, type=float, help='Weight decay if we apply some')

    return parser
