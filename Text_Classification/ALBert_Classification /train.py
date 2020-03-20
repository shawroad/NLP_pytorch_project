"""

@file  : train.py

@author: xiaolu

@time  : 2020-03-19

"""
import json
import torch
import random
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
from albert import Model
from pytorch_pretrained_bert.optimization import BertAdam
from dataloader import DatasetIterater
from config import Config


def evaluate(model, dev_data):
    '''
    测试
    :param model:
    :param test_data:
    :return:
    '''
    data_iter = DatasetIterater(dev_data, Config.batch_size, Config.device)

    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(data_iter)


def train(model):
    # (d, int(label), seq_len, mask)
    # 加载三种语料到一个列表中
    pos_data = json.load(open('./data/pos_data.json', 'r'))
    neu_data = json.load(open('./data/neu_data.json', 'r'))
    neg_data = json.load(open('./data/neg_data.json', 'r'))
    data = []
    data.extend(pos_data)
    data.extend(neu_data)
    data.extend(neg_data)

    random.shuffle(data)  # 打乱数据
    train_data = data[:900]
    dev_data = data[900:]

    train_iter = DatasetIterater(train_data, Config.batch_size, Config.device)

    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # 这里我们用bertAdam优化器
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=Config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * Config.num_epochs)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')

    model.train()
    for epoch in range(Config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, Config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(true, predic)  # 输入真实值和预测值 计算准确率
            print('epoch:{}, step:{}, loss:{}, accuracy:{}'.format(epoch, i, loss, train_acc))

            if total_batch % 30 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                dev_acc, dev_loss = evaluate(model, dev_data)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), Config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, {5}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, improve))
            model.train()
            total_batch += 1


if __name__ == '__main__':
    model = Model(Config).to(Config.device)
    train(model)
