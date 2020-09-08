"""

@file  : config.py

@author: xiaolu

@time  : 2020-01-03

"""
import logging
import torch


class Config:
    '''
    配置文件
    '''

    device = torch.device('cuda: 3' if torch.cuda.is_available() else 'cpu')

    # 数据和词典的位置
    data_file = 'data.pkl'
    vocab_file = 'vocab.pkl'

    pad_id = 0
    sos_id = 1
    eos_id = 2
    unk_id = 3
    IGNORE_ID = -1

    vocab_size = 1597

    grad_clip = 1.0  # clip gradients at an absolute value of

    print_freq = 10

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

