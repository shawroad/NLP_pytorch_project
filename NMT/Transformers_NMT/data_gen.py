"""

@file  : data_gen.py

@author: xiaolu

@time  : 2019-12-26

"""
import pickle
import time
import numpy as np
from torch.utils.data import Dataset
from config import logger
from config import Config
from utils import sequence_to_text
from torch.utils.data.dataloader import default_collate


logger.info('loading samples...')
# 加载数据　并统计了加载数据的时间
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
    '''
    是对序列进行padding
    :param batch: (padding后的输入, padding后的输出, 输入文本的真实长度)
    :return:
    '''
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
    def __init__(self, split):
        self.samples = data[split]

    def __getitem__(self, item):
        sample = self.samples[item]
        src_text = sample['in']   # 这里获取的数据是没有进行padding的, 对于输出 只是加了起始和结束标志
        tgt_text = sample['out']

        return np.array(src_text, dtype=np.long), np.array(tgt_text, np.long)

    def __len__(self):
        return len(self.samples)


def main():
    valid_dataset = TranslateDataset('valid')

    with open(Config.vocab_file, 'rb') as file:
        data = pickle.load(file)

    # 加载id转为文本 的词表
    src_idx2char = data['dict']['src_idx2char']
    tgt_idx2char = data['dict']['tgt_idx2char']

    src_text, tgt_text = valid_dataset[0]
    src_text = sequence_to_text(src_text, src_idx2char)  # 将id转为文本
    src_text = ' '.join(src_text)
    print('src_text: ' + src_text)

    tgt_text = sequence_to_text(tgt_text, tgt_idx2char)  # 将id转为文本
    tgt_text = ' '.join(tgt_text)
    print('tgt_text: ' + tgt_text)


if __name__ == "__main__":
    main()
