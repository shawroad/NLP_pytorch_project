"""

@file  : plot_loss.py

@author: xiaolu

@time  : 2020-01-03

"""
import time
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    '''
    从tar中提取模型 整理成pt文件
    # '''
    checkpoint = 'BEST_Model.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    loss = checkpoint['loss']
    plt.plot(loss)
    plt.show()