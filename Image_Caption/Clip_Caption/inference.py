"""
@file   : inference.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-30
"""
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer
from tqdm import tqdm
import os
from skimage import io
from PIL import Image
import argparse
from model import ClipCaptionModel
from os.path import join
import torch.nn.functional as F
import clip


def topk_filtering(logits, topk=10, topp=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 2  # batch size 1 for now - could be updated for more but the code would be less clear
    topk = min(topk, logits.size(-1))  # Safety check
    if topk > 0:
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        indices_to_remove = logits < torch.topk(logits, topk, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if topp > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > topp
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # todo check
        for i in range(sorted_indices_to_remove.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def generate(model, clip_embeds, tokenizer):
    b_size = clip_embeds.size(0)
    pad_id = tokenizer.pad_token_id
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id
    max_len = 100
    temperature = 1
    topk, topp = 5, 0.8

    cur_len = 0
    caption_ids = []    # 存储生成的caption

    # gpt2模型的输入: inputs_embeds:[bs, prefix_len, prefix_size]
    print(clip_embeds)
    print(type(clip_embeds))
    inputs_embeds = model.clip_project(clip_embeds).view(-1, model.prefix_len, model.prefix_size)
    finish_flag = [False] * b_size  # 第i个输入是否完成生成的标志

    while True:
        out = model.gpt2(inputs_embeds=inputs_embeds)
        logits = out.logits  # [b_size, len, vocab_size]
        next_token_logits = logits[:, -1, :]    # 取最后一个单词的预测分布
        next_token_logits = next_token_logits / temperature
        next_token_logits[:, unk_id] = -float('Inf')   # 将unk设为无穷小

        # topk filter
        filtered_logits = topk_filtering(next_token_logits, topk, topp)
        next_token_ids = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(1).tolist()

        # 分别判断生成图片是否已生成完毕
        for index in range(len(next_token_ids)):
            token_id = next_token_ids[index]
            # 如果第i个句子已经生成结束
            if finish_flag[index]:
                next_token_ids[index] = pad_id
            # 如果第i个句子生成结束
            elif token_id == sep_id:
                finish_flag[index] = True
            # 未结束生成
            elif cur_len == 0:
                caption_ids.append([token_id])
            else:
                caption_ids[index].append(token_id)
        next_token_ids = torch.tensor(next_token_ids).to(device)
        next_token_embeds = model.gpt2.transformer.wte(next_token_ids).to(device).unsqueeze(1)
        inputs_embeds = torch.cat((inputs_embeds, next_token_embeds), dim=1)

        cur_len += 1
        if cur_len > max_len or False not in finish_flag:
            break

    # 对token_id进行解码
    captions = []
    for caption_id in caption_ids:
        caption = tokenizer.convert_ids_to_tokens(caption_id)
        caption = ''.join(caption)
        captions.append(caption)

    return captions


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 分词器
    tokenizer = BertTokenizer.from_pretrained('./gpt2_pretrain')
    # BertMapper的模型配置
    bert_config = BertConfig.from_pretrained('./bert_pretrain')
    # 初始化模型
    model = ClipCaptionModel()
    if torch.cuda.is_available():
        model.to(device)

    # 加载权重
    model.load_state_dict(torch.load('./output/base_model_epoch1_step0.bin'))
    model.eval()
    model.half()

    # 加载clip模型
    clip_model, preprocess = clip.load('./clip_pretrain/ViT-B-32.pt', device=device, jit=False)

    # 加载数据集
    # img_path = '256063.jpg'
    img_path = '/usr/home/xiaolu10/xiaolu4/clip_caption/data/flickr30k-images/6609688031.jpg'
    image = io.imread(img_path)
    image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

    num_generate = 2
    clip_embeds = clip_model.encode_image(image)
    clip_embeds = clip_embeds.unsqueeze(1).repeat(1, num_generate, 1).view(-1, clip_embeds.size(-1))
    captions = generate(model, clip_embeds, tokenizer)
    print(captions)
