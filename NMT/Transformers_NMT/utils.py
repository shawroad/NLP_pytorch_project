"""

@file  : utils.py

@author: xiaolu

@time  : 2019-12-26

"""
import unicodedata
import re
import torch
import argparse


def encode_text(word_map, c):
    '''
    将文本映射成id
    :param word_map:
    :param c:
    :return:
    '''
    return [word_map.get(word, word_map['<unk>']) for word in c]


# 下面两个函数是对单词的清洗
def unicodeToAscii(s):
    '''
    unicode转Ascii
    :param s:
    :return:
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    '''
    单词清洗
    :param s:
    :return:
    '''
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def sequence_to_text(seq, idx2char):
    '''
    将id映射成文本
    :param seq:
    :param idx2char:
    :return:
    '''
    result = [idx2char[idx] for idx in seq]
    return result


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    '''
    保存模型
    :param epoch:
    :param epochs_since_improvement:
    :param model:
    :param optimizer:
    :param loss:
    :param is_best:
    :return:
    '''
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}

    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    进行梯度裁剪
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def parse_args():
    '''
    命令行参数
    :return:
    '''
    parser = argparse.ArgumentParser(description='Transformer')
    # Network architecture
    # encoder
    # TODO: automatically infer input dim
    parser.add_argument('--n_layers_enc', default=6, type=int,
                        help='Number of encoder stacks')
    parser.add_argument('--n_head', default=8, type=int,
                        help='Number of Multi Head Attention (MHA)')
    parser.add_argument('--d_k', default=64, type=int,
                        help='Dimension of key')
    parser.add_argument('--d_v', default=64, type=int,
                        help='Dimension of value')
    parser.add_argument('--d_model', default=512, type=int,
                        help='Dimension of model')
    parser.add_argument('--d_inner', default=2048, type=int,
                        help='Dimension of inner')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate')
    parser.add_argument('--pe_maxlen', default=5000, type=int,
                        help='Positional Encoding max len')
    # decoder
    parser.add_argument('--d_word_vec', default=512, type=int,
                        help='Dim of decoder embedding')
    parser.add_argument('--n_layers_dec', default=6, type=int,
                        help='Number of decoder stacks')
    parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int,
                        help='share decoder embedding with decoder projection')
    # Loss
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help='label smoothing')

    # Training config
    parser.add_argument('--epochs', default=1000, type=int,
                        help='Number of maximum epochs')
    # minibatch
    parser.add_argument('--shuffle', default=1, type=int,
                        help='reshuffle the data at every epoch')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--batch_frames', default=0, type=int,
                        help='Batch frames. If this is not 0, batch size will make no sense')
    parser.add_argument('--maxlen-in', default=50, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=25, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='Number of workers to generate minibatch')
    # optimizer
    parser.add_argument('--k', default=0.2, type=float,
                        help='tunable scalar multiply to learning rate')
    parser.add_argument('--warmup_steps', default=4000, type=int,
                        help='warmup steps')

    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()
    return args
