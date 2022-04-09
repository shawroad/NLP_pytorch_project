"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-30
"""
import torch
import torch.nn as nn
from typing import Tuple
from transformers import GPT2LMHeadModel, BertModel, BertConfig
from transformers.models.gpt2 import GPT2LMHeadModel, modeling_gpt2


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BertMapper(nn.Module):
    def __init__(self, bert_config, clip_size, prefix_size, prefix_len, constant_len):
        super(BertMapper, self).__init__()
        self.prefix_len = prefix_len
        self.prefix_size = prefix_size
        self.constant_len = constant_len
        self.bert = BertModel(config=bert_config)
        self.linear = nn.Linear(clip_size, prefix_len * prefix_size)
        self.prefix_const = nn.Parameter(torch.randn(constant_len, prefix_size), requires_grad=True)

    def forward(self, x):
        bs = x.size(0)
        # 将bs个图片向量映射成[bs, prefix_len, prefix_size]
        prefix = self.linear(x).view(-1, self.prefix_len, self.prefix_size)
        # [bs, constant_len, prefix_size]
        constant = self.prefix_const.unsqueeze(0).expand(bs, self.constant_len, self.prefix_size)
        # 将prefix向量与constant向量拼接，作为bert模型的输入
        prefix = torch.cat((prefix, constant), dim=1)
        # 输出捕获attention之后的prefix向量的输出
        out = self.bert(inputs_embeds=prefix)
        out = out.last_hidden_state[:, self.prefix_len:]
        return out


class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_len=10, clip_size=512, mapping_type='MLP', finetune_gpt2=False, constant_len=10):
        super(ClipCaptionModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('./gpt2_pretrain')

        # 如果没有预训练模型 自己初始化
        # model_config = modeling_gpt2.GPT2Config.from_json_file('./gpt2_pretrain/config.json')
        # self.gpt2 = GPT2LMHeadModel(config=model_config)

        # 将每个图片向量[clip_size] -> [prefix_len, prefix_size]
        self.prefix_size = self.gpt2.config.n_embd   # gpt2词嵌入的维度大小
        # print(self.prefix_size)   # 768

        self.prefix_len = prefix_len
        if mapping_type == 'MLP':
            self.clip_project = MLP((clip_size, (self.prefix_size * prefix_len) // 2, self.prefix_size * prefix_len))
        else:
            # BertMapper的模型配置
            bert_config = BertConfig.from_pretrained('./bert_pretrain/config.json')
            self.clip_project = BertMapper(bert_config, clip_size, self.prefix_size, prefix_len, constant_len)
        self.finetune_gpt2 = finetune_gpt2

    def forward(self, clip_embeds, caption_ids, mask):
        '''
        :param clip_embeds:  batch_size, clip_size
        :param caption_ids:  batch_size, max_len
        :param mask: batch_size, max_len
        :return:
        '''
        caption_embeds = self.gpt2.transformer.wte(caption_ids)
        # print(caption_embeds.size())   # torch.Size([2, 90, 768])

        prefix_embeds = self.clip_project(clip_embeds).view(-1, self.prefix_len, self.prefix_size)
        # print(prefix_embeds.size())    # torch.Size([2, 10, 768])

        embedding_cat = torch.cat((prefix_embeds, caption_embeds), dim=1)
        # print(embedding_cat.size())   # torch.Size([2, 100, 768])

        out = self.gpt2(inputs_embeds=embedding_cat, attention_mask=mask)
        return out.logits
