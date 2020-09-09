# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 10:18
# @Author  : xiaolu
# @FileName: train_distill.py
# @Software: PyCharm
import os
import torch
import pickle
import jieba
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from torch.autograd import Variable
from train_bert import BertClassification


def load_data(path):
    # 加载数据
    # os.path.join(args.checkpoint_path, "pytorch_model_epoch{}.bin".format(epoch))
    texts = []   # 这里的name主要指的是用那个数据去得到一个数据处理模型
    with open(os.path.join(path, 'train.txt'), encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = ' '.join(jieba.cut(line.split('\t', 1)[1].strip()))
            texts.append(line)
    # texts中放的是每一条数据  这些数据分词以后 用空格连着

    # 数据预处理
    tokenizer = Tokenizer(filters='', lower=True, split=' ', oov_token=1)
    tokenizer.fit_on_texts(texts)

    # 上面采用训练集训练一个数据处理的model  下面才是真正的数据处理
    # 训练集
    x_train, y_train = [], []
    text_train = []
    for line in open(os.path.join(path, 'train.txt'), encoding='utf8').readlines():
        label, text = line.split('\t', 1)
        text_train.append(text.strip())
        x_train.append(' '.join(jieba.cut(text.strip())))
        y_train.append(int(label))
    x_train = tokenizer.texts_to_sequences(x_train)

    # 验证集
    x_dev, y_dev = [], []
    text_dev = []
    for line in open(os.path.join(path, 'dev.txt'), encoding='utf8').readlines():
        label, text = line.split('\t', 1)
        text_dev.append(text.strip())
        x_dev.append(' '.join(jieba.cut(text.strip())))
        y_dev.append(int(label))
    x_dev = tokenizer.texts_to_sequences(x_dev)

    # 测试集
    x_test, y_test = [], []
    text_test = []
    for line in open(os.path.join(path, 'test.txt'), encoding="utf-8").readlines():
        label, text = line.split('\t', 1)
        text_test.append(text.strip())
        x_test.append(' '.join(jieba.cut(text.strip())))
        y_test.append(int(label))
    x_test = tokenizer.texts_to_sequences(x_test)
    v_size = len(tokenizer.word_index) + 1
    return (x_train, y_train, text_train), (x_dev, y_dev, text_dev), (x_test, y_test, text_test), v_size


class Teacher:
    def __init__(self, bert_model='bert_base_pretrain', max_seq=128):
        self.max_seq = max_seq
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True)
        self.model = BertClassification.from_pretrained(bert_model)
        self.model.load_state_dict(torch.load('./save_model/pytorch_model_epoch9.bin', map_location='cpu'))
        self.model.eval()

    def predict(self, text):
        # 输入一个文本  然后进行预测
        tokens = self.tokenizer.tokenize(text)[:self.max_seq]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)   # input_ids
        input_mask = [1] * len(input_ids)     # attention_mask
        padding = [0] * (self.max_seq - len(input_ids))     # padding
        input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
        input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)
        logits = self.model(input_ids, input_mask, None)
        return F.softmax(logits, dim=1).detach().cpu().numpy()


class MiniModel(nn.Module):
    def __init__(self, vocab_size):
        super(MiniModel, self).__init__()
        e_dim = 256
        h_dim = 256
        label_nums = 2
        self.emb = nn.Embedding(vocab_size, e_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(e_dim, h_dim, bidirectional=True, batch_first=True)   # 双向lstm
        self.fc = nn.Linear(h_dim * 2, label_nums)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, lens):
        embed = self.dropout(self.emb(x))
        out, _ = self.lstm(embed)
        hidden = self.fc(out[:, -1, :])
        return self.softmax(hidden), self.log_softmax(hidden)


def start_distill(args, model):
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        accu = []
        model.train()
        for i in range(0, len(x_train), args.batch_size):
            model.zero_grad()
            x = torch.LongTensor(x_train[i: i + args.batch_size]).to(device)
            y = torch.LongTensor(y_train[i: i + args.batch_size]).to(device)
            x_len = torch.LongTensor(l_train[i: i + args.batch_size]).to(device)
            distill_train = torch.FloatTensor(t_tr[i: i + args.batch_size]).to(device)
            py1, py2 = model(x, x_len)
            # 计算损失  重点就是这一步  loss1: 和硬标签的交叉熵  loss2: 和软标签的均方误差损失
            loss = args.alpha * ce_loss(py2, y) + (1 - args.alpha) * mse_loss(py1, distill_train)
            loss.backward()
            optimizer.step()
            print('[distill train starting] epoch: {}, step: {}, loss: {:10f}'.format(epoch, i//args.batch_size, loss))

        for i in range(0, len(x_dev), args.batch_size):
            model.zero_grad()
            x = torch.LongTensor(x_dev[i: i + args.batch_size]).to(device)
            y = torch.LongTensor(y_dev[i: i + args.batch_size]).to(device)
            x_len = torch.LongTensor(l_dev[i: i + args.batch_size]).to(device)
            distill_dev = torch.FloatTensor(t_de[i: i + args.batch_size]).to(device)
            py1, py2 = model(x, x_len)
            loss = mse_loss(py1, distill_dev)
            print('[distill dev starting] epoch: {}, step: {}, loss: {:10f}'.format(epoch, i//args.batch_size, loss))
            # 是否在验证集上也更新模型
            if args.teach_on_dev:
                loss.backward()
                optimizer.step()  # train only with teacher on dev set

        # 开始在测试集上测试了
        model.eval()
        with torch.no_grad():
            for i in range(0, len(x_test), args.batch_size):
                x = torch.LongTensor(x_test[i: i + args.batch_size])
                y = torch.LongTensor(y_test[i: i + args.batch_size])
                x_len = torch.LongTensor(l_test[i: i + args.batch_size])
                _, py = torch.max(model(x, x_len)[1], 1)
                accu.append((py == y).float().mean().item())
        print(np.mean(accu))

        # 保存蒸馏模型
        output_model_file = os.path.join(args.checkpoint_path, "distill_model_epoch{}.bin".format(epoch))
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        print("model save in %s" % output_model_file)
        torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == '__main__':
    # 开始蒸馏啦
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', )

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2, )
    parser.add_argument('--max_seq', type=int, default=128, )
    parser.add_argument('--learning_rate', type=float, default=0.002, help='this is learning of train')
    parser.add_argument('--device', type=str, default='0', help='this is learning of train')
    parser.add_argument('--is_need_knowledge', type=bool, default=False, help='you need to get knowledge')
    parser.add_argument('--alpha', type=float, default=0.5, help='you need to get knowledge')  # 两种损失的权重
    parser.add_argument('--teach_on_dev', type=bool, default=True, help='you need to get knowledge')  # 是否在验证集上也蒸馏
    parser.add_argument('--checkpoint_path', type=str, default='./save_model/', help='model save path')
    args = parser.parse_args()

    x_len = args.max_seq  # 最长
    (x_train, y_train, text_train), (x_dev, y_dev, text_dev), (x_test, y_test, text_test), v_size = load_data(args.data_dir)

    # 分别得出训练集, 验证集, 测试集每个样本的长度
    l_train = list(map(lambda x: min(len(x), x_len), x_train))
    l_dev = list(map(lambda x: min(len(x), x_len), x_dev))
    l_test = list(map(lambda x: min(len(x), x_len), x_test))

    x_train = sequence.pad_sequences(x_train, maxlen=x_len)
    x_dev = sequence.pad_sequences(x_dev, maxlen=x_len)
    x_test = sequence.pad_sequences(x_test, maxlen=x_len)
    # print(x_train.shape)   # (999, 128)
    # print(x_dev.shape)     # (8000, 128)
    # print(x_test.shape)    # (1000, 128)
    device = torch.device('cuda: {}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    if args.is_need_knowledge:
        # 需要用老师模型去推理一遍数据
        teacher = Teacher()
        print(teacher.predict('还不错！这个价位算是物有所值了！'))  # [[0.01699478 0.9830052 ]]
        with torch.no_grad():
            # 对训练集和验证集进行预测
            t_tr = np.vstack([teacher.predict(text) for text in tqdm(text_train)])
            t_de = np.vstack([teacher.predict(text) for text in tqdm(text_dev)])

        with open('./knowledge/train', 'wb') as fout:
            pickle.dump(t_tr, fout)

        with open('./knowledge/dev', 'wb') as fout:
            pickle.dump(t_de, fout)
    else:
        # 之前有推理过 所以这里直接加载
        t_tr = pickle.load(open('./knowledge/train', 'rb'))
        t_de = pickle.load(open('./knowledge/dev', 'rb'))

    # 现在构造一个小模型
    ce_loss = nn.NLLLoss()
    mse_loss = nn.MSELoss()
    model = MiniModel(v_size)
    start_distill(args, model)
