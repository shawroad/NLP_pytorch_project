"""
# -*- coding: utf-8 -*-
# @File    : train.py
# @Time    : 2020/12/8 5:24 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import json
import numpy as np
from random import choice
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import time
from config import set_args
from model import s_model, po_model


def get_now_time():
    # 获取当前时间
    return time.ctime(time.time())


def seq_padding(X):
    # 对所有序列进行padding
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]


def seq_padding_vec(X):
    L = [len(x) for x in X]
    ML = max(L)
    # print("ML",ML)
    return [x + [[1, 0]] * (ML - len(x)) for x in X]


class Data_Generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size   # 一轮的总步数
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def pro_res(self):
        idxs = list(range(len(self.data)))   # 对每条数据指定索引
        np.random.shuffle(idxs)
        T, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], []
        for i in idxs:
            d = self.data[i]
            text = d['text']   # 文本
            # 遍历关系
            items = {}
            for sp in d['spo_list']:
                subjected = text.find(sp[0])
                objected = text.find(sp[2])
                if subjected != -1 and objected != -1:
                    key = (subjected, subjected + len(sp[0]))   # 主题的起始和终止做为key
                    if key not in items:
                        items[key] = []
                    items[key].append((objected, objected + len(sp[2]), predicate2id[sp[1]]))
            if items:
                T.append([char2id.get(c, 1) for c in text])  # 1: unk   0:padding
                s1, s2 = [0] * len(text), [0] * len(text)
                # s1中1位置 是每个主体的起始位置
                # s2中1位置 是每个主体的结束位置
                for j in items:
                    # 默认遍历字典中的键   是subject实体的起始和结束
                    s1[j[0]] = 1
                    s2[j[1] - 1] = 1

                k1, k2 = choice(list(items.keys()))    # 一个句子可能对应多对关系 我们随机抽取一个关系
                o1, o2 = [0] * len(text), [0] * len(text)  # 0是unk类（共49+1个类）
                # print(items)  # {(0, 8): [(31, 35, 28), (23, 26, 28), (9, 11, 48), (27, 30, 28)]}
                for j in items[(k1, k2)]:    # 遍历抽到的这对关系
                    o1[j[0]] = j[2]   # 客体:关系
                    o2[j[1] - 1] = j[2]   # 客体: 关系
                S1.append(s1)
                S2.append(s2)
                K1.append([k1])
                K2.append([k2 - 1])
                O1.append(o1)
                O2.append(o2)
                # print(s1)
                # print(s2)
                # print(k1)
                # print(k2-1)
                # print(o1)
                # print(o2)
        T = np.array(seq_padding(T))
        S1 = np.array(seq_padding(S1))
        S2 = np.array(seq_padding(S2))
        O1 = np.array(seq_padding(O1))
        O2 = np.array(seq_padding(O2))
        K1, K2 = np.array(K1), np.array(K2)
        return [T, S1, S2, K1, K2, O1, O2]


class MyDataset(Dataset):
    def __init__(self, _T, _S1, _S2, _K1, _K2, _O1, _O2):
        self.x_data = _T
        self.y1_data = _S1
        self.y2_data = _S2
        self.k1_data = _K1
        self.k2_data = _K2
        self.o1_data = _O1
        self.o2_data = _O2
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y1_data[index], self.y2_data[index], self.k1_data[index], self.k2_data[index], self.o1_data[index], self.o2_data[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    t = np.array([item[0] for item in data], np.int32)
    s1 = np.array([item[1] for item in data], np.int32)
    s2 = np.array([item[2] for item in data], np.int32)
    k1 = np.array([item[3] for item in data], np.int32)

    k2 = np.array([item[4] for item in data], np.int32)
    o1 = np.array([item[5] for item in data], np.int32)
    o2 = np.array([item[6] for item in data], np.int32)
    return {
        'T': torch.LongTensor(t),  # targets_i
        'S1': torch.FloatTensor(s1),
        'S2': torch.FloatTensor(s2),
        'K1': torch.LongTensor(k1),
        'K2': torch.LongTensor(k2),
        'O1': torch.LongTensor(o1),
        'O2': torch.LongTensor(o2),
    }


def extract_items(text_in):
    R = []
    _s = [char2id.get(c, 1) for c in text_in]
    _s = np.array([_s])
    _k1, _k2, t, t_max, mask = s_model(torch.LongTensor(_s))
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _kk1s = []
    for i, _kk1 in enumerate(_k1):
        if _kk1 > 0.5:
            _subject = ''
            for j, _kk2 in enumerate(_k2[i:]):
                if _kk2 > 0.5:
                    _subject = text_in[i: i + j + 1]
                    break
            if _subject:
                _k1, _k2 = torch.LongTensor([[i]]), torch.LongTensor([[i + j]])  # np.array([i]), np.array([i+j])
                _o1, _o2 = po_model(t.cuda(), t_max.cuda(), _k1.cuda(), _k2.cuda())
                _o1, _o2 = _o1.cpu().data.numpy(), _o2.cpu().data.numpy()
                _o1, _o2 = np.argmax(_o1[0], 1), np.argmax(_o2[0], 1)

                for i, _oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for j, _oo2 in enumerate(_o2[i:]):
                            if _oo2 == _oo1:
                                _object = text_in[i: i + j + 1]
                                _predicate = id2predicate[_oo1]
                                # print((_subject, _predicate, _object))
                                R.append((_subject, _predicate, _object))
                                break
        _kk1s.append(_kk1.data.cpu().numpy())
    _kk1s = np.array(_kk1s)
    return list(set(R))


def evaluate():
    A, B, C = 1e-10, 1e-10, 1e-10
    cnt = 0
    for d in tqdm(iter(dev_data)):
        R = set(extract_items(d['text']))
        T = set([tuple(i) for i in d['spo_list']])
        A += len(R & T)
        B += len(R)
        C += len(T)
        cnt += 1
    return 2 * A / (B + C), A / B, A / C


if __name__ == '__main__':
    args = set_args()
    # 加载数据
    train_data = json.load(open('./data/train_data_me.json'))
    dev_data = json.load(open('./data/dev_data_me.json'))
    id2predicate, predicate2id = json.load(open('./data/all_50_schemas_me.json'))
    id2predicate = {int(i): j for i, j in id2predicate.items()}
    id2char, char2id = json.load(open('./data/all_chars_me.json'))
    num_classes = len(id2predicate)

    # 生成数据
    dg = Data_Generator(train_data)
    T, S1, S2, K1, K2, O1, O2 = dg.pro_res()

    # print('数据量:', len(T))    # 数据量: 21
    torch_dataset = MyDataset(T, S1, S2, K1, K2, O1, O2)

    data_loader = DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,   # subprocesses for loading data
    )

    s_model = s_model(len(char2id) + 2, args.char_size, args.hidden_size)
    po_model = po_model(len(char2id) + 2, args.char_size, args.hidden_size, 49)

    params = list(s_model.parameters())
    params += list(po_model.parameters())

    optimizer = torch.optim.Adam(params, lr=0.001)

    cross_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    best_f1 = 0
    best_epoch = 0

    for epoch in range(args.num_epochs):
        for step, loader_res in tqdm(iter(enumerate(data_loader))):
            t_s = loader_res["T"]
            k1 = loader_res["K1"]
            k2 = loader_res["K2"]
            s1 = loader_res["S1"]
            s2 = loader_res["S2"]
            o1 = loader_res["O1"]
            o2 = loader_res["O2"]
            # print(t_s.size())   # torch.Size([2, 126])
            # print(s1.size())    # torch.Size([2, 126])
            # print(k1.size())    # torch.Size([2, 1])
            # print(o1.size())    # torch.Size([2, 126])
            ps_1, ps_2, t, t_max, mask = s_model(t_s)
            # print(ps_1.size())   # torch.Size([2, 126, 1])
            # print(ps_2.size())   # torch.Size([2, 126, 1])
            # print(t.size())    # torch.Size([2, 126, 128])
            # print(t_max.size())   # torch.Size([2, 128])
            # print(mask.size())    # torch.Size([2, 126, 1])
            po_1, po_2 = po_model(t, t_max, k1, k2)
            # print(po_1.size())    # torch.Size([2, 126, 50])
            # print(po_2.size())    # torch.Size([2, 126, 50])
            s1 = torch.unsqueeze(s1, 2)
            s2 = torch.unsqueeze(s2, 2)

            # 预测主体的开始和结束的位置损失
            s1_loss = bce_loss(ps_1, s1)
            s1_loss = torch.sum(torch.mul(s1_loss, mask)) / torch.sum(mask)
            s2_loss = bce_loss(ps_2, s2)
            s2_loss = torch.sum(torch.mul(s2_loss, mask)) / torch.sum(mask)

            po_1 = po_1.permute(0, 2, 1)   # torch.Size([2, 50, 126])
            po_2 = po_2.permute(0, 2, 1)   # # torch.Size([2, 50, 126])

            o1_loss = cross_loss(po_1, o1)
            o1_loss = torch.sum(torch.mul(o1_loss, mask[:, :, 0])) / torch.sum(mask)
            o2_loss = cross_loss(po_2, o2)
            o2_loss = torch.sum(torch.mul(o2_loss, mask[:, :, 0])) / torch.sum(mask)

            loss_sum = 2.5 * (s1_loss + s2_loss) + (o1_loss + o2_loss)
            print('epoch: {}, step:{}, loss: {:10f}'.format(epoch, step, loss_sum))
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
        torch.save(s_model.state_dict(), 'save_model/s_' + str(epoch) + '.bin')
        torch.save(po_model.state_dict(), 'save_model/po_' + str(epoch) + '.bin')
        f1, precision, recall = evaluate()
        print("epoch:", epoch, "loss:", loss_sum.data)

        if f1 >= best_f1:
            best_f1 = f1
            best_epoch = epoch

        print('f1: %.4f, precision: %.4f, recall: %.4f, bestf1: %.4f, bestepoch: %d \n ' % (f1, precision, recall, best_f1, best_epoch))
