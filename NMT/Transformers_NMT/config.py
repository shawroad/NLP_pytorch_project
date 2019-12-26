"""

@file  : config.py

@author: xiaolu

@time  : 2019-12-26

"""
import logging
import torch


class Config:
    # 指定设备
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    # 整理好数据要放置的位置
    data_file = './data/data.pkl'   # 将输入和输出整理
    vocab_file = './data/vocab.pkl'   # 词典要放的位置

    # 训练集所放的位置
    train_translation_zh_filename = './data/train.zh'
    train_translation_en_filename = './data/train.en'

    # 验证集所放的位置
    valid_translation_zh_filename = './data/valid.zh'
    valid_translation_en_filename = './data/valid.en'

    # 输入词表和输出词表的大小
    n_src_vocab = 15000
    n_tgt_vocab = 15000

    # 几种特殊标志
    pad_id = 0
    sos_id = 1
    eos_id = 2
    unk_id = 3
    IGNORE_ID = -1

    # 主要是为了过滤一些过长的文本
    maxlen_in = 50
    maxlen_out = 100

    # 保存断点
    checkpoint = None  # path to checkpoint, None if none

    # 梯度裁剪的范围
    grad_clip = 1.0  # clip gradients at an absolute value of 保持梯度始终在-1到1之间

    # 每隔50步打印一次
    print_freq = 50  # print training/validation stats  every __ batches
    d_model = 512  # 隐层维度大小


def get_logger():
    '''
    打印日志
    :return:
    '''
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()

