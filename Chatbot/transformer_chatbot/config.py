"""

@file   : config.py

@author : xiaolu

@time   : 2019-12-26

"""
import logging
import torch


class Config:
    # 指定设备
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    # 数据所在的文件夹
    train_filename = './data/12万对话语料青云库.csv'

    # 字典写入的位置  以及整理好数据保存的位置
    vocab_file = './data/vocab.pkl'
    data_file = './data/data.pkl'

    # 其实结束标志
    pad_id = 0
    sos_id = 1
    eos_id = 2
    unk_id = 3
    IGNORE_ID = -1

    maxlen_in = 50
    maxlen_out = 50

    # 词典的大小
    vocab_size = 5884

    grad_clip = 1.0  # clip gradients at an absolute value of (-1, 1)

    print_freq = 50

    checkpoint = None

    d_model = 512


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()