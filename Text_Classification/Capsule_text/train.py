"""

@file  : train.py

@author: xiaolu

@time  : 2020-06-04

"""
import json
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score
from rlog import _log_normal, _log_warning, _log_info, _log_error, _log_toomuch, _log_bg_blue, _log_bg_pp, _log_fg_yl, \
    _log_fg_cy, _log_black, rainbow
from capsule import Capsule_Main
from config import Config


def evaluate(sentence_id_dev, labels_dev, lengths_dev, model):
    sentence_id_dev = torch.LongTensor(sentence_id_dev).to(Config.device)
    labels_dev = torch.LongTensor(labels_dev).to(Config.device)

    with torch.no_grad():
        logits = model(sentence_id_dev)

        pred = np.argmax(logits.cpu().data.numpy(), axis=1)
        acc_dev = accuracy_score(labels_dev.cpu().data.numpy(), pred)

        loss_dev = loss_func(logits, labels_dev)
    return acc_dev, loss_dev


if __name__ == '__main__':
    data = json.load(open('./data/train.json', 'r'))
    sentence_ids = data['sentence_ids']
    labels = data['labels']
    lengths = data['sentence_len']

    # 准备少量的验证集
    sentence_id_dev = sentence_ids[:1000]
    labels_dev = labels[:1000]
    lengths_dev = lengths[:1000]

    # print(len(sentence_ids))
    # print(len(labels))
    # print(len(lengths))
    learning_rate = 0.001
    model = Capsule_Main()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(Config.device)
    loss_func = nn.CrossEntropyLoss()  # 用二分类方法预测是否属于该类，而非多分类

    data_nums = len(labels)
    step_nums = data_nums // Config.batch_size

    acc_best = 0

    for epoch in range(1, Config.Epoch+1):
        for i in range(step_nums-1):
            sentence_id = torch.LongTensor(sentence_ids[i*Config.batch_size: (i+1)*Config.batch_size]).to(Config.device)
            label = torch.LongTensor(labels[i*Config.batch_size: (i+1)*Config.batch_size]).to(Config.device)
            length = torch.LongTensor(lengths[i*Config.batch_size: (i+1)*Config.batch_size]).to(Config.device)
            # print(sentence_id.size())   # torch.Size([2, 225])
            # print(label.size())  # torch.Size([2])
            # print(length.size())  # torch.Size([2])
            logits = model(sentence_id)
            # print(logits.size())  # torch.Size([16, 2])

            pred = np.argmax(logits.cpu().data.numpy(), axis=1)
            acc = accuracy_score(label.cpu().data.numpy(), pred)

            loss = loss_func(logits, label)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            out = 'Epoch:{}, steps:{}, loss:{:6f}, accuracy:{:6f}'.format(epoch, i, loss, acc)
            rainbow(out)
        if epoch % 2 == 0:
            acc_dev, loss_dev = evaluate(sentence_id_dev, labels_dev, lengths_dev, model)
            print("epoch:{}, dev_loss:{:6f}, dev_acc:{:6f}".format(epoch, loss_dev, acc_dev))
            if acc_dev > acc_best:
                acc_best = acc_dev
                torch.save(model.state_dict(), './best_model.bin')
                model.train()
