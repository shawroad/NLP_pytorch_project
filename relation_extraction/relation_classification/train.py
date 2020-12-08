"""
# -*- coding: utf-8 -*-
# @File    : train.py
# @Time    : 2020/12/8 11:39 上午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import os
import pickle
import gzip
import torch
from torch import optim
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import recall_score, precision_score, accuracy_score
from BiLSTM_ATT import BiLSTM_ATT
from config import set_args



class Example(object):
    def __init__(self, text, position1, position2, label):
        self.text = text
        self.position1 = position1
        self.position2 = position2
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "text: %s" % (self.text)
        s += ", position1: %s" % (self.position1)
        s += ", position2: %s" % (self.position2)
        s += ", label: %s" % (self.label)
        return s


class Features(object):
    def __init__(self, input_ids, position1, position2, label_id):
        self.input_ids = input_ids
        self.position1 = position1
        self.position2 = position2
        self.label_id = label_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "input_ids: %s" % (self.input_ids)
        s += ", position1: %s" % (self.position1)
        s += ", position2: %s" % (self.position2)
        s += ", label_id: %s" % (self.label_id)
        return s


def evaluate(eval_features):
    print("***** Running evaluating *****")
    print("  Num examples = {}".format(len(eval_features)))
    print("  Batch size = {}".format(args.eval_batch_size))
    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_position1 = torch.tensor([f.position1 for f in eval_features], dtype=torch.long)
    eval_position2 = torch.tensor([f.position2 for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(eval_input_ids, eval_position1, eval_position2, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss = 0
    step = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for input_ids, position1, position2, label_ids in tqdm(eval_dataloader, desc='Evaluation'):
        step += 1
        input_ids = input_ids.to(device)
        position1 = position1.to(device)
        position2 = position2.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, position1, position2)
            loss = loss_func(logits, label_ids)
        eval_loss += loss.mean().item()  # 统计一个batch的损失 一个累加下去
        labels = label_ids.data.cpu().numpy()
        predic = torch.max(logits.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

    # 损失 召回率 查准率
    eval_loss = eval_loss / step
    eval_accuracy = accuracy_score(labels_all, predict_all)
    s = 'epoch:{}, eval_loss: {}, eval_accuracy:{}'.format(epoch, eval_loss, eval_accuracy)
    print(s)
    s += '\n'
    with open('result_eval.txt', 'a+') as f:
        f.write(s)
    return eval_loss, eval_accuracy


if __name__ == '__main__':
    args = set_args()
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    # 加载训练集
    with gzip.open(args.train_features_path, 'rb') as f:
        train_features = pickle.load(f)

    with gzip.open(args.dev_features_path, 'rb') as f:
        eval_features = pickle.load(f)

    model = BiLSTM_ATT()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # 交叉熵损失
    loss_func = nn.CrossEntropyLoss()

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_position1 = torch.tensor([f.position1 for f in train_features], dtype=torch.long)
    all_position2 = torch.tensor([f.position2 for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([int(f.label_id) for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_position1, all_position2, all_label_ids)
    model.train()
    model.to(device)
    best_loss = None

    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_features)))
    print("  Batch size = {}".format(args.train_batch_size))
    for epoch in range(args.num_train_epochs):
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input_ids, position1, position2, labels_ids = batch
            logits = model(input_ids, position1, position2)
            loss = loss_func(logits, labels_ids)
            print('epoch:{}, step:{}, loss:{:10f}'.format(epoch, step, loss))
            loss.backward()
            optimizer.step()

        # 验证验证集
        test_loss, test_acc = evaluate(eval_features)
        # 验证训练集中四万之后的数据
        model.train()

        if best_loss is None or best_loss > test_loss:
            best_loss = test_loss
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            os.makedirs(args.save_model, exist_ok=True)
            output_model_file = os.path.join(args.save_model, "best_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.save_model, "epoch{}_ckpt.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)




