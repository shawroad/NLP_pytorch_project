# -*- coding: utf-8 -*-
"""
@Time ： 2020/10/30 9:54
@Auth ： xiaolu
@File ：config.py
@IDE ：PyCharm
@Email：luxiaonlp@163.com
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/atis", type=str, help="The input data dir")
    parser.add_argument("--model_dir", default='bert_pretrain', required=False, type=str, help="Path to save, load model")
    parser.add_argument('--seed', type=int, default=43, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=2, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int, help="Batch size for evaluation.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument("--save_model", default='save_model', required=False, type=str, help="Path to save, load model")
    parser.add_argument("--use_crf", default='True', type=bool, help="Whether to use CRF")   # 是否使用CRF
    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')
    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    args = parser.parse_args()
    return args



    # parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    # parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    # parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")
    #
    # parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    #
    #
    # parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    # parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    # parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    # parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    # parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    #
    # parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    # parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
    #
    # parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    # parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    # parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    #

    # # CRF option
    # parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    # parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")
    #
    # args = parser.parse_args()
    # return args
    # # args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
