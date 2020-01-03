"""

@file   : run.py

@author : xiaolu

@time   : 2019-12-31

"""
import time
import torch
import numpy as np
import ERNIE
from utils import build_dataset, build_iterator, get_time_dif
from train_eval import train, init_network


if __name__ == '__main__':
    dataset = 'THUCNews'   # 数据集
    config = ERNIE.Config(dataset)  # 导入数据集

    # 为了让模型每次结果运行一致  我们给定随机种子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    time_diff = get_time_dif(start_time)

    # train
    model = ERNIE.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)