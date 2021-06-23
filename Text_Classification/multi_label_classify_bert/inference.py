"""
@file   : inference.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-23
"""
import torch
from tqdm import tqdm
import pandas as pd
from model import Model
from config import set_args
from torch.utils.data import Dataset, DataLoader
from transformers.models.bert import BertTokenizer


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
            tags = [int(x) - 1 for x in tags]  # 标签是从零开始的
            label = torch.zeros((args.num_classes,))
            label[tags] = 1  # 转成类似one_hot标签
            return input_ids, attention_mask, label


def test_model():
    result = []
    model.eval()
    tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
    with torch.no_grad():
        for idx, (input_ids, attention_mask) in enumerate(tk):
            input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
            output = model(input_ids, attention_mask)

            for res in output:  # 后处理，找大于0.5的类别（阈值可以微调），如果多了就取TOP5，如果没有就取TOP1
                _, res1 = torch.topk(res, 5)
                res1 = res1.cpu().numpy()

                res2 = torch.where(res > 0.5)[0]
                res2 = res2.cpu().numpy()

                if len(res2) > 5:
                    result.append(res1)
                elif len(res2) == 0:
                    result.append(res1[0])
                else:
                    result.append(res2)

    with open('submission.csv', 'w')as f:
        for i in range(len(result)):
            f.write(str(i) + ',')
            res = [str(x + 1) for x in result[i]]
            if len(res) < 5:
                res += ['-1'] * (5 - len(res))
            f.write(','.join(res))
            f.write('\n')


if __name__ == '__main__':
    args = set_args()
    test = pd.read_csv(args.test_data)
    test_set = MyDataset(test, test=True)
    tokenizer = BertTokenizer.from_pretrained(args.vocab)

    model = Model()

    # 加载权重
    model.load_state_dict(torch.load('model_epoch1.bin'))

    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)
    test_model()
