"""

@file   : train_bidaf.py

@author : xiaolu

@time   : 2020-02-16

"""
import argparse
import numpy as np
import json
import torch
import random
from torch.utils.data.dataset import Dataset
from torch import nn
from torch import optim
from math import log2
import torch.nn.functional as F
from tqdm import tqdm
import json
import re
from collections import Counter
import string
import os
from BiDAF import BiDAF
from config import Config


class SQuADDataset(Dataset):
    '''
    数据generator 每次取出一个batch的数据
    '''
    def __init__(self, npz_file, num_steps, batch_size):
        super().__init__()
        # 1. 加载预处理后的语料
        data = np.load(npz_file)

        # 读取文章预处理数据
        self.context_idxs = torch.from_numpy(data["context_idxs"]).long()
        self.context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long()

        # 读取问题预处理数据
        self.ques_idxs = torch.from_numpy(data["ques_idxs"]).long()
        self.ques_char_idxs = torch.from_numpy(data["ques_char_idxs"]).long()

        # 读取答案的起始和结束标志
        self.y1s = torch.from_numpy(data["y1s"]).long()
        self.y2s = torch.from_numpy(data["y2s"]).long()
        self.ids = torch.from_numpy(data["ids"]).long()

        num = len(self.ids)
        self.batch_size = batch_size

        # 直接指定步数  或者我们算出步数(所有数据量除以batch_size)即可得出。
        self.num_steps = num_steps if num_steps >= 0 else num // batch_size
        num_items = num_steps * batch_size   # 一个epoch 我们需要这么多数据 (如果的步数过多，则有些数据可能需要重复)
        idxs = list(range(num))
        self.idx_map = []
        i, j = 0, num

        while j <= num_items:
            random.shuffle(idxs)
            self.idx_map += idxs.copy()
            i = j
            j += num
        random.shuffle(idxs)
        self.idx_map += idxs[:num_items - i]   # 随机的数据标号 [4, 3, 43, 2, 54, 65, 34...]

    def __len__(self):
        return self.num_steps

    def __getitem__(self, item):
        idxs = torch.LongTensor(self.idx_map[item:item + self.batch_size])
        res = (self.context_idxs[idxs],
               self.context_char_idxs[idxs],
               self.ques_idxs[idxs],
               self.ques_char_idxs[idxs],
               self.y1s[idxs],
               self.y2s[idxs], self.ids[idxs])
        return res


class EMA(object):
    '''
    EMA（Exponential Moving Average）是指数移动平均值
    '''
    def __init__(self, decay):
        self.decay = decay
        self.shadows = {}
        self.devices = {}

    def __len__(self):
        return len(self.shadows)

    def get(self, name: str):
        return self.shadows[name].to(self.devices[name])

    def set(self, name: str, param: nn.Parameter):
        self.shadows[name] = param.data.to('cpu').clone()
        self.devices[name] = param.data.device

    def update_parameter(self, name: str, param: nn.Parameter):
        if name in self.shadows:
            data = param.data
            new_shadow = self.decay * data + (1.0 - self.decay) * self.get(name)
            param.data.copy_(new_shadow)
            self.shadows[name] = new_shadow.to('cpu').clone()


def valid(model, dataset, eval_file):
    '''
    验证数据集
    :param model:
    :param dataset:
    :param eval_file:
    :return:
    '''
    model.eval()
    answer_dict = {}
    losses = []
    num_batches = Config.val_num_batches
    with torch.no_grad():
        for i in tqdm(random.sample(range(0, len(dataset)), num_batches), total=num_batches):
            Cwid, Ccid, Qwid, Qcid, y1, y2, ids = dataset[i]
            Cwid, Ccid, Qwid, Qcid = Cwid.to(Config.device), Ccid.to(Config.device), Qwid.to(Config.device), Qcid.to(Config.device)
            p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
            y1, y2 = y1.to(Config.device), y2.to(Config.device)
            loss1 = F.nll_loss(p1, y1, reduction='mean')
            loss2 = F.nll_loss(p2, y2, reduction='mean')
            loss = (loss1 + loss2) / 2
            losses.append(loss.item())
            yp1 = torch.argmax(p1, 1)
            yp2 = torch.argmax(p2, 1)
            yps = torch.stack([yp1, yp2], dim=1)
            ymin, _ = torch.min(yps, 1)
            ymax, _ = torch.max(yps, 1)
            answer_dict_, _ = convert_tokens(eval_file, ids.tolist(), ymin.tolist(), ymax.tolist())
            answer_dict.update(answer_dict_)
            print(loss)

    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    print("VALID loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"], metrics["exact_match"]))


def test(model, dataset, eval_file):
    model.eval()
    answer_dict = {}
    losses = []
    num_batches = Config.test_num_batches
    with torch.no_grad():
        for i in tqdm(range(num_batches), total=num_batches):
            Cwid, Ccid, Qwid, Qcid, y1, y2, ids = dataset[i]
            Cwid, Ccid, Qwid, Qcid = Cwid.to(Config.device), Ccid.to(Config.device), Qwid.to(Config.device), Qcid.to(Config.device)
            p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
            y1, y2 = y1.to(Config.device), y2.to(Config.device)
            loss1 = F.nll_loss(p1, y1, reduction='mean')
            loss2 = F.nll_loss(p2, y2, reduction='mean')
            loss = (loss1 + loss2) / 2
            losses.append(loss.item())
            yp1 = torch.argmax(p1, 1)
            yp2 = torch.argmax(p2, 1)
            yps = torch.stack([yp1, yp2], dim=1)
            ymin, _ = torch.min(yps, 1)
            ymax, _ = torch.max(yps, 1)
            answer_dict_, _ = convert_tokens(eval_file, ids.tolist(), ymin.tolist(), ymax.tolist())
            answer_dict.update(answer_dict_)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    f = open("log/answers.json", "w")
    json.dump(answer_dict, f)
    f.close()
    metrics["loss"] = loss
    print("TEST loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"], metrics["exact_match"]))
    return metrics


def convert_tokens(eval_file, qa_id, pp1, pp2):
    '''
    找出起始和结束　然后输出答案那一截
    :param eval_file:
    :param qa_id:
    :param pp1:
    :param pp2:
    :return:
    '''
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        l = len(spans)
        if p1 >= l or p2 >= l:
            ans = ""
        else:
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            ans = context[start_idx: end_idx]
        answer_dict[str(qid)] = ans
        remapped_dict[uuid] = ans
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def train(model, optimizer, scheduler, ema, dataset, start, length):
    '''
    真正的训练
    :param model: 模型
    :param optimizer: 优化器
    :param scheduler: 学习率调整
    :param ema: 指数平滑移动
    :param dataset: 数据集
    :param start:
    :param length:
    :return:
    '''
    model.train()
    losses = []
    # for i in tqdm(range(start, length + start), total=length):
    for i in range(start, length + start):
        optimizer.zero_grad()
        # print(dataset[0])
        # print(dataset[i])
        Cwid, Ccid, Qwid, Qcid, y1, y2, ids = dataset[i]
        # print(Cwid.size())   # torch.Size([2, 400])
        # print(Ccid.size())   # torch.Size([2, 400, 16])
        # print(Qwid.size())   # torch.Size([2, 50])
        # print(Qcid.size())   # torch.Size([2, 50, 16])
        # print(y1.size())    # torch.Size([2])
        # print(y2.size())    # torch.Size([2])
        # exit()

        Cwid, Ccid, Qwid, Qcid = Cwid.to(Config.device), Ccid.to(Config.device), Qwid.to(Config.device), Qcid.to(Config.device)

        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)

        y1, y2 = y1.to(Config.device), y2.to(Config.device)

        # 计算损失
        loss1 = F.nll_loss(p1, y1, reduction='mean')
        loss2 = F.nll_loss(p2, y2, reduction='mean')
        loss = (loss1 + loss2) / 2
        losses.append(loss.item())
        loss.backward()

        optimizer.step()
        scheduler.step()

        for name, p in model.named_parameters():
            if p.requires_grad:
                ema.update_parameter(name, p)  # 梯度平滑指数更新
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)   # 梯度裁剪
        print('step:{}, loss: {:8f}'.format(i, loss))

    loss_avg = np.mean(losses)
    print("STEP {:8d} loss {:8f}\n".format(i + 1, loss_avg))


def train_entry(args):
    '''
    训练
    :return:
    '''

    # 加载词向量 字符向量 训练集中的验证集  测试集中的验证集
    with open(Config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(Config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(Config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(Config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    print("Building model...")

    train_dataset = SQuADDataset(Config.train_record_file, Config.num_steps, Config.batch_size)
    dev_dataset = SQuADDataset(Config.dev_record_file, Config.test_num_batches, Config.batch_size)

    lr = Config.learning_rate
    base_lr = 1.0
    warm_up = Config.lr_warm_up_num

    model = BiDAF(word_mat, char_mat).to(Config.device)
    ema = EMA(Config.ema_decay)  # 指数平均移动

    # 参数更新使用指数平滑
    for name, p in model.named_parameters():
        if p.requires_grad:
            ema.set(name, p)

    # 取出模型中的所有参数 然后去定义优化器
    params = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adam(lr=base_lr, betas=(Config.beta1, Config.beta2), eps=1e-7, weight_decay=3e-7, params=params)

    # 学习率进行调整
    cr = lr / log2(warm_up)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: cr * log2(ee + 1) if ee < warm_up else lr)

    L = Config.checkpoint
    N = Config.num_steps

    # 两种评价指标
    best_f1 = best_em = patience = 0
    for iter in range(0, N, L):
        train(model, optimizer, scheduler, ema, train_dataset, iter, L)

        valid(model, train_dataset, train_eval_file)
        # 测试数据
        metrics = test(model, dev_dataset, dev_eval_file)
        print("Learning rate: {}".format(scheduler.get_lr()))
        dev_f1 = metrics["f1"]
        dev_em = metrics["exact_match"]
        if dev_f1 < best_f1 and dev_em < best_em:
            patience += 1
            if patience > Config.early_stop:
                break
        else:
            patience = 0
            best_f1 = max(best_f1, dev_f1)
            best_em = max(best_em, dev_em)

        fn = os.path.join(Config.save_dir, "model.pt")
        torch.save(model, fn)


if __name__ == '__main__':
    print("Current device is {}".format(Config.device))

    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--dev-file', default='dev-v1.1.json')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--word-dim', default=100, type=int)
    args = parser.parse_args()

    train_entry(args)


