import os
import torch
import pandas as pd
import skimage.io as io
import clip
from PIL import Image
import pickle
import argparse
from tqdm import tqdm, trange
from os.path import join
# from loguru import logger


def main():
    device = torch.device('cuda:0')
    
    clip_model, preprocess = clip.load('../clip_pretrain/ViT-B-32.pt', device=device, jit=False)
    
    image_path = ''
    df = pd.read_csv('./corpus/filckr_caption.csv')
    
    base_dir = './corpus/flickr30k-images/'
    files = os.listdir(base_dir)
    
    image_id2embed = {}
    for img_name in tqdm(list(set(df['image_id'].tolist()))):
        # image_id                caption
        img_path = base_dir + img_name        
        try:
            image = io.imread(img_path)
        except:
            continue
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_embed = clip_model.encode_image(image).cpu()
        image_id2embed[img_name] = clip_embed
    
    pickle.dump(image_id2embed, open('flickr30k_image_id2embed.pkl', 'wb'))


if __name__ == '__main__':
    main()
