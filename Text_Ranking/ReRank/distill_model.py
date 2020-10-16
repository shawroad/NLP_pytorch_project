# -*- coding: utf-8 -*-
# @Time    : 2020/9/29 9:33
# @Author  : xiaolu
# @FileName: Distill_Model.py
# @Software: PyCharm
import torch
from torch import nn
from transformers import BertLayer
from transformers import BertModel
from transformers import BertConfig


class BertEmbeddings(nn.Module):
    # bert词嵌入部分
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CModel(nn.Module):
    def __init__(self, device):
        super(CModel, self).__init__()
        self.device = device
        self.num_labels = 2
        self.config = BertConfig.from_pretrained('./roberta_pretrain/bert_config.json')
        self.embeddings = BertEmbeddings(self.config)

        num_layers = 3
        self.layer = nn.ModuleList([BertLayer(self.config) for _ in range(num_layers)])
        self.output = nn.Linear(self.config.hidden_size, self.num_labels)   # 分类

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        input_shape = input_ids.size()
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        # print(embedding_output.size())    # torch.Size([2, 512, 768])

        # 对attention_mask进行处理
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 进行三层bert的计算
        layer_3_output = []
        hidden_states = embedding_output
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]
            layer_3_output.append(hidden_states)

        output_state = hidden_states[:, 0]
        # print(output.size())   # torch.Size([2, 768])
        logits = self.output(output_state)
        logits = logits.softmax(dim=1)
        return logits, layer_3_output

        