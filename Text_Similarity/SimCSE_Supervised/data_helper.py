"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-04
"""
import json
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source):
        origin = source['origin']
        entailment = source['entailment']
        contradiction = source['contradiction']
        sample = self.tokenizer([origin, entailment, contradiction],
                                max_length=self.maxlen,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])


class TestDataset:
    def __init__(self, data, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.traget_idxs = self.text_to_id([x[0] for x in data])
        self.source_idxs = self.text_to_id([x[1] for x in data])
        self.label_list = [int(x[2]) for x in data]
        assert len(self.traget_idxs['input_ids']) == len(self.source_idxs['input_ids'])

    def text_to_id(self, source):
        sample = self.tokenizer(source, max_length=self.maxlen, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def get_data(self):
        return self.traget_idxs, self.source_idxs, self.label_list


def load_data(path):
    data = []
    with open(path) as f:
        for i in f:
            data.append(json.loads(i))
    return data


def load_STS_data(path):
    data = []
    with open(path) as f:
        for i in f:
            d = i.split("||")
            sentence1 = d[1]
            sentence2 = d[2]
            score = int(d[3])
            data.append([sentence1, sentence2, score])
    return data


if __name__ == '__main__':
    path = './data/cnsd_snli_v1.0.train.jsonl'
    load_data(path)

