"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-25
"""
import json
import os
import torch
import numpy as np
import jieba
from config import set_args
from scipy.misc import imread, imresize
from torch.utils.data import Dataset


args = set_args()


def encode_caption(word_map, c):
    return [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
        word_map['<end>']] + [word_map['<pad>']] * (args.max_len - len(c))


class CaptionDataset(Dataset):
    def __init__(self, mode, transform=None, word_map=None):
        self.mode = mode
        self.word_map = word_map
        self.transform = transform
        assert self.mode in {'train', 'valid'}
        
        if mode == 'train':
            json_path = '/usr/home/xiaolu10/xiaolu4/Image_Captioning/data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json'
            self.image_folder = '/usr/home/xiaolu10/xiaolu4/Image_Captioning/data/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
            
        else:
            json_path = '/usr/home/xiaolu10/xiaolu4/Image_Captioning/data/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json'
            self.image_folder = '/usr/home/xiaolu10/xiaolu4/Image_Captioning/data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/'
            
        self.samples = json.load(open(json_path, 'r', encoding='utf8'))

        # 每张图片有五个描述  因此总的数据集个数为: 样本数 * 5
        self.captions_per_image = 5   # 每个图片有5个描述

        self.dataset_size = len(self.samples * self.captions_per_image)

    def __getitem__(self, i):
        sample = self.samples[i // self.captions_per_image]
        path = os.path.join(self.image_folder, sample['image_id'])   # 图片的路径

        # 1. 处理图片
        img = imread(path)   # 读取图片   读出来的size=(H, W, channel_num)
        if len(img.shape) == 2:
            # 如果是两通道  直接加一维  弄成三通道
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)

        img = imresize(img, (256, 256))   # 将图片变为256  为了适配后面的resnet模型
        img = img.transpose(2, 0, 1)   # 将(H, W, channel_num) 变为(channel_num, W, H)
        assert img.shape == (3, 256, 256)
        assert np.max(img) <= 255

        img = torch.tensor(img / 255., dtype=torch.float)   # 归一化

        if self.transform is not None:
            img = self.transform(img)

        # 处理文本
        captions = sample['caption']
        assert len(captions) == self.captions_per_image
        c = captions[i % self.captions_per_image]
        c = list(jieba.cut(c))
        enc_c = encode_caption(self.word_map, c)

        caption = torch.tensor(enc_c, dtype=torch.long)
        caplen = torch.tensor([len(c) + 2], dtype=torch.long)

        if self.mode is 'train':
            return img, caption, caplen
        else:
            all_captions = torch.LongTensor([encode_caption(self.word_map, list(jieba.cut(c))) for c in captions])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
