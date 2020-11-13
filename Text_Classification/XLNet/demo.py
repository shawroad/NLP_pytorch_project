# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/13 15:04
@Auth ： xiaolu
@File ：test.py
@IDE ：PyCharm
@Email：luxiaonlp@163.com
"""
import torch
from pdb import set_trace
from transformers import XLNetConfig, XLNetModel
from transformers import XLNetTokenizer


if __name__ == '__main__':
    tokenizer = XLNetTokenizer.from_pretrained('./xlnet_pretrain/spiece.model')
    config = XLNetConfig.from_pretrained('./xlnet_pretrain/config.json')
    model = XLNetModel.from_pretrained('./xlnet_pretrain/pytorch_model.bin', config=config)

    text = '你是我患得患失的梦' * 500
    text = tokenizer.tokenize(text)
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(text)).view(1, -1)
    print(input_ids.size())
    output = model(input_ids)
    set_trace()


