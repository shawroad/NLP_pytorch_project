"""

@file  : train.py

@author: xiaolu

@time  : 2019-10-28

"""
from utils import *
from model import *
from config import Config
import sys
import torch
import torch.optim as optim
from torch import nn


if __name__ == '__main__':
    config = Config()
    train_file = './data/ag_news.train'

    if len(sys.argv) > 2:
        train_file = sys.argv[1]

    test_file = './data/ag_news.test'

    if len(sys.argv) > 3:
        test_file = sys.argv[2]

    # 得到三种数据集的迭代器
    train_iterator, test_iterator, val_iterator = get_iterators(config, train_file, test_file)

    # 创建具体的模型, 优化器和损失函数
    model = CharCNN(config)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    # 向模型中加入损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(loss_fn)

    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = model.run_epoch(train_iterator, val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, train_iterator)
    val_acc = evaluate_model(model, val_iterator)
    test_acc = evaluate_model(model, test_iterator)

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))