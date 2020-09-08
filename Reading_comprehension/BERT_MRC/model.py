"""

@file  : RNet.py

@author: xiaolu

@time  : 2020-03-04

"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert import BertModel, BertTokenizer
from config import Config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 加载预训练模型
        self.bert = BertModel.from_pretrained(Config.model_bert_path)
        self.qa_outputs = nn.Linear(768, 2)
        self.loss_fct = CrossEntropyLoss()   # 计算损失

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        # _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)

        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            start_loss = self.loss_fct(start_logits, start_positions)
            end_loss = self.loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2  # !!!
            return total_loss, start_logits, end_logits

        else:
            return start_logits, end_logits

