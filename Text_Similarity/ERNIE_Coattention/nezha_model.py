"""
@file   : nezha_model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-16
"""
import torch
from torch import nn
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

        self.all_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size*3, mid_size),
                nn.BatchNorm1d(mid_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(mid_size, 2)
            )
            for _ in range(self.task_num)
        ])

    def forward(self, source_input_ids, target_input_ids):
        source_attention_mask = torch.ne(source_input_ids, 0)   # 等于0则为False否则为True
        target_attention_mask = torch.ne(target_input_ids, 0)

        # 得到nezha的输出
        source_embedding = self.nezha_model(source_input_ids, attention_mask=source_attention_mask)
        target_embedding = self.nezha_model(target_input_ids, attention_mask=target_attention_mask)

        # 取出cls
        source_embedding = source_embedding[1]
        target_embedding = target_embedding[1]
        # print(source_embedding.size())   # batch_size, hidden_size
        # print(target_embedding.size())

        abs_embedding = torch.abs(source_embedding - target_embedding)
        context_embedding = torch.cat([source_embedding, target_embedding, abs_embedding], -1)
        context_embedding = self.dropout(context_embedding)

        # 接个全连接
        all_probs = [classifier(context_embedding) for classifier in self.all_classifier]
        return all_probs

