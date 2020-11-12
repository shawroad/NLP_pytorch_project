# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/12 15:59
@Auth ： xiaolu
@File ：test2.py
@IDE ：PyCharm
@Email：luxiaonlp@163.com
"""
import torch
from pdb import set_trace
from transformers import BertTokenizer, AdamW
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size


config = LongformerConfig.from_pretrained('./longformer_pretrain')
config.attention_mode = 'sliding_chunks'
model = Longformer.from_pretrained('./longformer_pretrain', config=config)

tokenizer = BertTokenizer.from_pretrained('./longformer_pretrain/vocab.txt')
tokenizer.model_max_length = model.config.max_position_embeddings

input_text = '你是我患得患失的梦' * 200
input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

print(input_ids.size())

input_ids, attention_mask = pad_to_window_size(
    input_ids, attention_mask, config.attention_window[0], tokenizer.pad_token_id)
print(input_ids.size())

attention_mask[:, 0] = 2  # global attention for the first token
# print(input_ids.tolist())
# print(attention_mask.tolist())

output = model(input_ids, attention_mask=attention_mask)
# len(output) = 2
# output[0].size()   # torch.Size([1, 2048, 768])
# output[1].size()   # torch.Size([1, 768])
set_trace()




