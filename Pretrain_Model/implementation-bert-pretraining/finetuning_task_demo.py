# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 19:50
# @Author  : xiaolu
# @FileName: finetuning_task_demo.py
# @Software: PyCharm
import torch
from transformers import BertConfig, BertModel
from transformers import BertTokenizer


config = BertConfig.from_pretrained('./corpus/config.json')
model = BertModel.from_pretrained('./corpus/pytorch_model_epoch50.bin', config=config)
tokenizer = BertTokenizer.from_pretrained('./corpus/vocab.txt')
temp = tokenizer.encode_plus('你是煞笔', '你是刹车差', pad_to_max_length=True, max_length=128)
input_ids = torch.LongTensor([temp['input_ids']])
segment_ids = torch.LongTensor([temp['token_type_ids']])
attention_mask = torch.LongTensor([temp['attention_mask']])
seq_output, _ = model(input_ids, segment_ids, attention_mask)
print(seq_output.size())


