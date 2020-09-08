# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 10:51
# @Author  : xiaolu
# @FileName: data_loader.py
# @Software: PyCharm
from torch.utils.data.dataset import Dataset
import json
import numpy as np
import torch
from collections import namedtuple


InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids
    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1
    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids
    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class Data_pretrain(Dataset):
    def __init__(self, data_path, tokenizer):
        seq_len = 256
        with open(data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            num_samples = len(lines)
            # 加载训练样本
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)

            for i, line in enumerate(lines):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)

                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.input_ids = input_ids
            self.input_masks = input_masks
            self.segment_ids = segment_ids
            self.lm_label_ids = lm_label_ids
            self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)))


if __name__ == '__main__':
    #  training_path, file_id, tokenizer, data_name, reduce_memory=False
    from transformers import BertTokenizer
    from torch.utils.data.dataloader import DataLoader
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain/vocab.txt')
    train_data_path = './process_data0.json'
    txt = Data_pretrain(train_data_path, tokenizer)

    data_iter = DataLoader(txt, shuffle=True, batch_size=2)
    for batch in data_iter:
        print(batch[0])
        print(batch[1])
        print(batch[2])
        print(batch[3])
        print(batch[4])
        print(batch[0].size())
        print(batch[1].size())
        print(batch[2].size())
        print(batch[3].size())
        print(batch[4].size())
        exit()



