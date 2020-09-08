"""

@file  : inference.py

@author: xiaolu

@time  : 2020-01-03

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
    filename = 'reading_comprehension.pt'  # 导出模型所放的位置

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
        samples = data
    elapsed = time.time() - start
    logger.info('elapsed: {:.4f} seconds'.format(elapsed))

    # 加载词典
    logger.info('loading vocab...')
    start = time.time()
    with open(Config.vocab_file, 'rb') as file:
        data = pickle.load(file)
        id2vocab = data['id2vocab']
    elapsed = time.time() - start
    logger.info('elapsed: {:.4f} seconds'.format(elapsed))

    for i in [0, 50, 100, 200]:
        input_id = samples['input_corpus'][i]
        output_id = samples['input_corpus'][i]

        input_sent = torch.from_numpy(np.array(input_id, dtype=np.long)).to(Config.device)
        input_len = torch.LongTensor([len(output_id)]).to(Config.device)
        output_sent = torch.from_numpy(np.array(output_id, dtype=np.long)).to(Config.device)

        with torch.no_grad():
            nbest_hyps = model.recognize(input=input_sent, input_length=input_len, char_list=id2vocab)

        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [id2vocab[idx] for idx in out]
            out = ''.join(out)
            out = out.replace('<sos>', '').replace('<eos>', '')
            print("最后的预测:", out)

