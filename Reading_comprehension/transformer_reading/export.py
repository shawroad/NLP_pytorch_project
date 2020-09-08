"""

@file  : export.py

@author: xiaolu

@time  : 2020-01-03

"""
import time
import torch
from transformer.transformer import Transformer


if __name__ == '__main__':
    '''
    从tar中提取模型 整理成pt文件
    '''
    checkpoint = 'BEST_Model.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model']
    print(type(model))

    filename = 'reading_comprehension.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))

    print('loading {}...'.format(filename))
    start = time.time()
    model = Transformer()
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))
