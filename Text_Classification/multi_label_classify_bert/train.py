"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-22
"""
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from model import Model
from config import set_args
from utils import AverageMeter
from transformers.models.bert import BertTokenizer
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, dataframe, maxlen=256, test=False):
        self.df = dataframe
        self.maxlen = maxlen
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 将问题和其对应的细节进行拼接
        text = str(self.df.question_title.values[idx]) + str(self.df.question_detail.values[idx])
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=self.maxlen, return_tensors='pt')

        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]

        if self.test:
            return input_ids, attention_mask
        else:
            # 如果不是测试集  制作标签
            tags = self.df.tag_ids.values[idx].split('|')
            tags = [int(x) - 1 for x in tags]   # 标签是从零开始的
            label = torch.zeros((args.num_classes,))
            label[tags] = 1   # 转成类似one_hot标签
            return input_ids, attention_mask, label


def train_model(model, train_loader):
    model.train()
    losses = AverageMeter()

    optimizer.zero_grad()
    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    for idx, (input_ids, attention_mask, y) in enumerate(tk):

        if torch.cuda.is_available():
            input_ids, attention_mask, y = input_ids.cuda(), attention_mask.cuda(), y.cuda()

        output = model(input_ids, attention_mask)
        # print(output.size())    # torch.Size([16, 25551])

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), y.size(0))
        tk.set_postfix(loss=losses.avg)
    return losses.avg


if __name__ == '__main__':
    args = set_args()
    train = pd.read_csv(args.train_data)

    tokenizer = BertTokenizer.from_pretrained(args.vocab)

    train_set = MyDataset(train)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)

    model = Model()

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(args.num_epoch):
        train_loss = train_model(model, train_loader)
        # 一轮训练完后保存一下
        torch.save(model.state_dict(), 'model_epoch{}.bin'.format(epoch))

