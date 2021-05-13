"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-11
"""
import os
import torch
from torch import nn
from model import Model
from config import set_args
from torch.utils.data import Dataset, DataLoader


class Lyric_Dataset(Dataset):
    def __init__(self):
        self.dataset = []
        with open('./data/processed_data.txt', 'r', encoding='utf8') as f:
            words = [int(word) for word in f.read().split()]
            words_length = len(words)
            start = 0
            while words_length - start > args.pos_num + 1:
                self.dataset.append(words[start: start + args.pos_num + 1])
                start += args.stride
            else:
                if words_length > args.pos_num + 1:
                    self.dataset.append(words[words_length - args.pos_num - 1:])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = torch.tensor(self.dataset[item])
        return data[0: -1], data[1:]   # [1, 2, 3, 4] => [1, 2, 3], [2, 3, 4]


if __name__ == '__main__':
    args = set_args()
    train_dataloader = DataLoader(Lyric_Dataset(), batch_size=3, shuffle=True)

    model = Model()

    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fct = nn.CrossEntropyLoss()
    loss_new = 10000000
    for epoch in range(args.num_train_epoch):
        for step, batch in enumerate(train_dataloader):
            data, label = batch
            position = torch.arange(0, data.shape[1]).repeat(data.shape[0], 1)
            # print(position.size())   # batch_size, max_len

            if torch.cuda.is_available():
                data, position, label = data.cuda(), position.cuda(), label.cuda()
            output = model(data, position)
            # print(output.size())     # torch.Size([3, 200, 1032])
            # print(label.size())     # torch.Size([3, 200])

            loss = loss_fct(output.view(-1, output.size(-1)), label.view(-1))
            print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, loss))

            opt.zero_grad()
            loss.backward()
            opt.step()

            if loss.item() < loss_new:
                loss_new = loss.item()
                os.makedirs(args.save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.save_path, 'pytorch_model.bin'))

