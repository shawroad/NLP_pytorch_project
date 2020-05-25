"""

@file  : data_loader.py

@author: xiaolu

@time  : 2020-05-25

"""
import random
import numpy as np
import os
import torch
from transformers import BertTokenizer
from config import Config


class DataLoader:
    def __init__(self):
        # 1. 加载标签  标签到id的映射
        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        self.tokenizer = BertTokenizer.from_pretrained(Config.model_vocab_path)
        self.tag_pad_idx = -1   # padding的时候　就直接将标签padding成-1
        self.token_pad_idx = 0

    def load_tags(self):
        # 加载标签
        tags = []
        file_path = os.path.join(Config.base_path, 'tags.txt')
        with open(file_path, 'r') as f:
            for tag in f:
                tags.append(tag.strip())
        return tags

    def load_sentences_tags(self, sentences_file, tags_file, d):
        # 加载句子以及对应的标签
        sentences = []
        tags = []
        with open(sentences_file, 'r') as file:
            for line in file:
                tokens = self.tokenizer.tokenize(line.strip())
                sentences.append(self.tokenizer.convert_tokens_to_ids(tokens))

        with open(tags_file, 'r') as file:
            for line in file:
                tag_seq = [self.tag2idx.get(tag) for tag in line.strip().split(' ')]
                tags.append(tag_seq)

        assert len(sentences) == len(tags)
        for i in range(len(sentences)):
            assert len(tags[i]) == len(sentences[i])

        d['data'] = sentences
        d['tags'] = tags
        d['size'] = len(sentences)

    def load_data(self, data_type):
        # 加载三种数据集
        data = {}

        if data_type in ['train', 'val', 'test']:
            sentences_file = os.path.join(Config.base_path, data_type, 'sentences.txt')
            tags_path = os.path.join(Config.base_path, data_type, 'tags.txt')
            self.load_sentences_tags(sentences_file, tags_path, data)
        return data

    def data_iterator(self, data, shuffle=False):
        order = list(range(data['size']))
        if shuffle:
            random.seed(2020)
            random.shuffle(order)

        for i in range(data['size'] // Config.batch_size):
            sentences = [data['data'][idx] for idx in order[i * Config.batch_size: (i+1) * Config.batch_size]]
            tags = [data['tags'][idx] for idx in order[i * Config.batch_size: (i+1) * Config.batch_size]]
            batch_len = len(sentences)

            # padding
            batch_max_len = max(len(s) for s in sentences)
            max_len = min(batch_max_len, Config.max_len)   # 一个是批量最小, 一个是整个数据集最长

            batch_data = self.token_pad_idx * np.ones((batch_len, max_len))
            batch_tags = self.tag_pad_idx * np.ones((batch_len, max_len))
            for j in range(batch_len):
                cur_len = len(sentences[j])
                if cur_len <= max_len:
                    batch_data[j][:cur_len] = sentences[j]
                    batch_tags[j][:cur_len] = tags[j]
                else:
                    batch_data[j] = sentences[j][:max_len]
                    batch_tags[j] = tags[j][:max_len]

            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_tags = torch.tensor(batch_tags, dtype=torch.long)
            batch_data, batch_tags = batch_data.to(Config.device), batch_tags.to(Config.device)
            yield batch_data, batch_tags

