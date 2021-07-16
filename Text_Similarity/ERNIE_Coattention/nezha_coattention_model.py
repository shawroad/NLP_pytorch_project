"""
@file   : nezha_coattention_model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-16
"""
import torch
import math
from torch import nn
from pdb import set_trace
from NEZHA.model_nezha import NEZHAModel, NezhaConfig
from NEZHA import nezha_utils


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.nezha_config = NezhaConfig.from_json_file('./model_weight/NEZHA/config.json')
        self.nezha_model = NEZHAModel(config=self.nezha_config)
        nezha_utils.torch_init_model(self.nezha_model, './model_weight/NEZHA/pytorch_model.bin')

        hidden_size = 768
        mid_size = 512
        self.task_num = 2   # 相当于并行两个全连接 然后得出两个概率 分别为 a类数据集 和b类数据集的预测  即:多任务

        self.dropout = nn.Dropout(0.5)
        self.co_attention = CoAttention(config=self.nezha_config)   # 相当于再加了一层transformer的encoder

        self.all_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 3, mid_size),
                nn.BatchNorm1d(mid_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(mid_size, 2)
            )
            for _ in range(self.task_num)
        ])

    def forward(self, source_input_ids, target_input_ids):
        source_attention_mask = torch.ne(source_input_ids, 0)  # size: batch_size, max_len
        target_attention_mask = torch.ne(target_input_ids, 0)

        # get bert output
        source_embedding = self.nezha_model(source_input_ids, attention_mask=source_attention_mask)
        target_embedding = self.nezha_model(target_input_ids, attention_mask=target_attention_mask)
        # print(source_embedding[0].size())   # batch_size, max_len, hidden_size
        # print(source_embedding[1].size())   # batch_size, hidden_size

        source_coattention_outputs = self.co_attention(target_embedding[0], source_embedding[0], source_attention_mask)
        target_coattention_outputs = self.co_attention(source_embedding[0], target_embedding[0], target_attention_mask)
        source_coattention_embedding = source_coattention_outputs[:, 0, :]
        target_coattention_embedding = target_coattention_outputs[:, 0, :]

        abs_embedding = torch.abs(source_coattention_embedding - target_coattention_embedding)
        context_embedding = torch.cat([source_coattention_embedding, target_coattention_embedding, abs_embedding], -1)
        context_embedding = self.dropout(context_embedding)

        all_probs = [classifier(context_embedding) for classifier in self.all_classifier]
        return all_probs


class CoAttention(nn.Module):
    def __init__(self, config):
        super(CoAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size   # head_num * head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # x: batch_size, max_len, head_num * head_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # batch_size, max_len, head_num, head_size
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, context_states, query_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        mixed_query_layer = self.query(query_states)

        # batch_size, max_len ->  batch_size, 1, 1, max_len
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.float()  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0   # mask位置搞成负无穷
        attention_mask = extended_attention_mask

        mixed_key_layer = self.key(context_states)    # torch.Size([2, 152, head_num * head_size])
        mixed_value_layer = self.value(context_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask   # 给mask位置 搞成负无穷

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        outputs = context_layer.view(*new_context_layer_shape)
        return outputs
