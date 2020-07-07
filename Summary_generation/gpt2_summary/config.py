# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 14:05
# @Author  : xiaolu
# @FileName: config.py
# @Software: PyCharm
import torch


class Config:
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    model_output_path = './summary_model'   # 创建对话模型的输出目录
    pretrained_model = False    # 预训练模型的路径  若没有 赋值False

    gpt2_config = './gpt2_model/model_config_dialogue_small.json'
    gpt2_vocab = './gpt2_model/vocab_small.txt'

    train_raw_path = './data/train_with_summary.txt'   # 原始语料的路径
    train_tokenized_path = './data/train_tokenized.txt'   # 将语料转为id 存的路径

    batch_size = 2

    epochs = 5
    gradient_accumulation = 2
    lr = 1.5e-4
    warmup_steps = 2000

    writer_dir = './tensorboard_summary'   # 日志输出文件

    max_grad_norm = 1.0

    log_step = 2





