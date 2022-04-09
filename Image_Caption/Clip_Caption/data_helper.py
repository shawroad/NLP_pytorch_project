"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-30
"""
import torch
import pandas as pd
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, BertTokenizer
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from os.path import join
import glob
import skimage.io as io
from PIL import Image


class ClipCapDataset(Dataset):
    def __init__(self, prefix_len, tokenizer, max_len):
        self.normalize_prefix = False   # 是否对prefix的向量归一化
        pad_id = tokenizer.pad_token_id
        df = pd.read_csv('./data/corpus/all_caption.csv')
        image_id2embed = pickle.load(open('./data/corpus/all_imageid2embed.pkl', 'rb'))
        clip_embeds = []
        caption_ids_list = []
        mask_list = []

        for image_id, caption in zip(df['image_id'].tolist(), df['caption'].tolist()):
            clip_embed = image_id2embed[image_id].squeeze(0).float()
            caption_ids = tokenizer.encode(caption, add_special_tokens=False)
            caption_ids.append(tokenizer.sep_token_id)

            # 对超长caption_ids截断
            caption_ids = caption_ids[:(max_len - prefix_len)]
            mask = [1] * (prefix_len + len(caption_ids))

            # padding
            padding_len = max_len - prefix_len - len(caption_ids)
            caption_ids += [pad_id] * padding_len
            mask += [0] * padding_len

            caption_ids = torch.tensor(caption_ids, dtype=torch.long)
            mask = torch.tensor(mask, dtype=torch.long)

            clip_embeds.append(clip_embed)
            caption_ids_list.append(caption_ids)
            mask_list.append(mask)
        self.clip_embeds = clip_embeds
        self.caption_ids_list = caption_ids_list
        self.mask_list = mask_list

    def __len__(self):
        return len(self.caption_ids_list)

    def __getitem__(self, index: int):
        clip_embed = self.clip_embeds[index]
        caption_ids = self.caption_ids_list[index]
        mask = self.mask_list[index]
        if self.normalize_prefix:
            clip_embed = clip_embed / clip_embed.norm(2, -1)    # todo check
        return clip_embed, caption_ids, mask


class ImageDataset(Dataset):
    def __init__(self, path, preprocess):
        # 加载路径下的所有图片
        self.images = []
        self.image_names = []
        for file in glob.glob(join(path, '*')):
            image = io.imread(file)
            image = preprocess(Image.fromarray(image)).squeeze(0)
            filename = os.path.basename(file)
            self.images.append(image)
            self.image_names.append(filename)

    def __getitem__(self, item):
        return self.images[item], self.image_names[item]

    def __len__(self) -> int:
        return len(self.images)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('./gpt2_pretrain/vocab.txt')
    prefix_len = 10
    max_len = 100

    clip = ClipCapDataset(prefix_len, tokenizer, max_len)

