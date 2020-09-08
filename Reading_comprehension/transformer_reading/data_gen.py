"""

@file  : data_gen.py

@author: xiaolu

@time  : 2020-01-03

"""
import pickle
import time
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from config import Config, logger
from data_process import sequence_to_text

# 加载数据
logger.info('loading samples...')
start = time.time()
with open(Config.data_file, 'rb') as file:
    data = pickle.load(file)
elapsed = time.time() - start
logger.info('elapsed: {:.4f}'.format(elapsed))


def get_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    data = [line.strip() for line in data]
    return data


def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        src, tgt = elem
        # 找出这一批数据中最长文本的长度
        max_input_len = max_input_len if max_input_len > len(src) else len(src)
        max_target_len = max_target_len if max_target_len > len(tgt) else len(tgt)

    for i, elem in enumerate(batch):
        src, tgt = elem
        input_length = len(src)
        padded_input = np.pad(src, (0, max_input_len - len(src)), 'constant', constant_values=Config.pad_id)
        padded_target = np.pad(tgt, (0, max_target_len - len(tgt)), 'constant', constant_values=Config.IGNORE_ID)
        batch[i] = (padded_input, padded_target, input_length)
        # 这三部分分别是: padding后输入, padding后的输出, input_length输入的真实长度

    # sort it by input lengths (long to short)
    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


class TranslateDataset(Dataset):
    def __init__(self):
        self.samples = data

    def __getitem__(self, item):
        src_text = self.samples['input_corpus'][item]
        tgt_text = self.samples['output_corpus'][item]

        return np.array(src_text, dtype=np.long), np.array(tgt_text, np.long)

    def __len__(self):
        return len(self.samples['input_corpus'])


def main():
    train_dataset = TranslateDataset()

    with open(Config.vocab_file, 'rb') as file:
        data = pickle.load(file)

    # 加载id转为文本 的词表
    src_idx2char = data['id2vocab']
    tgt_idx2char = data['id2vocab']

    src_text, tgt_text = train_dataset[0]
    src_text = sequence_to_text(src_text, src_idx2char)  # 将id转为文本
    src_text = ''.join(src_text)
    print('src_text: ' + src_text)

    tgt_text = sequence_to_text(tgt_text, tgt_idx2char)  # 将id转为文本
    tgt_text = ''.join(tgt_text)
    print('tgt_text: ' + tgt_text)


if __name__ == "__main__":
    main()
