"""

@file  : chat.py

@author: xiaolu

@time  : 2019-12-27

"""
import pickle
import random
import time

import numpy as np
import torch

from config import Config, logger
from transformer.transformer import Transformer

if __name__ == '__main__':
    # 先去执行export.py 把模型导出来
    filename = 'chatbot-v2.pt'  # 导出模型所放的位置

    print('loading {}...'.format(filename))
    start = time.time()

    model = Transformer()
    model.load_state_dict(torch.load(filename))

    print('elapsed {} sec'.format(time.time() - start))
    model = model.to(Config.device)
    model.eval()

    # 加载测试集
    logger.info('loading samples...')
    start = time.time()
    with open(Config.data_file, 'rb') as file:
        data = pickle.load(file)
        samples = data['test']
    elapsed = time.time() - start
    logger.info('elapsed: {:.4f} seconds'.format(elapsed))

    # 加载词典
    logger.info('loading vocab...')
    start = time.time()
    with open(Config.vocab_file, 'rb') as file:
        data = pickle.load(file)
        idx2char = data['dict']['idx2char']
        char2idx = data['dict']['char2idx']
    elapsed = time.time() - start
    logger.info('elapsed: {:.4f} seconds'.format(elapsed))

    # samples = random.sample(samples, 10)  # 从测试集中抽取10条
    print("想退出请安q")
    sign = 1
    while sign != 'q':
        string = input("我:")
        char = list(string)
        sentenct_in = [char2idx.get(c, Config.unk_id) for c in char]

        input_sent = torch.from_numpy(np.array(sentence_in, dtype=np.long)).to(Config.device)
        input_length = torch.LongTensor([len(sentence_in)]).to(Config.device)

        sentence_in = ''.join([idx2char[idx] for idx in sentence_in])

        with torch.no_grad():
            nbest_hyps = model.recognize(input=input_sent, input_length=input_length, char_list=idx2char)

        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [idx2char[idx] for idx in out]
            out = ''.join(out)
            out = out.replace('<sos>', '').replace('<eos>', '')

            print('机:', out)

