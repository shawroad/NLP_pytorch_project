"""

@file  : albert.py

@author: xiaolu

@time  : 2020-03-19

"""
from transformers import AlbertModel, AlbertConfig
from torch import nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = AlbertConfig.from_pretrained(config.albert_config_path)
        self.albert = AlbertModel.from_pretrained(config.albert_model_path, config=self.config)
        for param in self.albert.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        _, pooled = self.albert(context, attention_mask=mask)
        out = self.fc(pooled)
        return out