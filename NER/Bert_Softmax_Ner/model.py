"""

@file  : model.py

@author: xiaolu

@time  : 2020-05-25

"""
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig
import torch.nn.functional as F
from config import Config


class BertSoftmaxForNer(nn.Module):
    def __init__(self):
        super(BertSoftmaxForNer, self).__init__()
        # 加载预训练模型
        num_tag = Config.num_tag
        self.config = BertConfig.from_pretrained(Config.model_config_path)
        self.bert = BertModel.from_pretrained(Config.model_path, config=self.config)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_tag)
        self.loss_fct = CrossEntropyLoss(ignore_index=0)   # 计算损失

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # print(sequence_output.size())    # torch.Size([2, 52, 768])
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # print(logits.size())   # torch.Size([2, 52, 7])
        if labels is not None:
            active_logits = logits.view(-1, Config.num_tag)
            active_logits = F.softmax(active_logits, dim=1)
            active_labels = labels.view(-1)
            loss = self.loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits

