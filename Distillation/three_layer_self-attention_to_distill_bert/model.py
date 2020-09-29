# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 9:33
# @Author  : xiaolu
# @FileName: model.py
# @Software: PyCharm
from torch import nn
from transformers import BertModel, BertConfig
from torch.nn import CrossEntropyLoss


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./roberta_pretrain/bert_config.json')
        self.roberta = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.config)
        self.num_labels = 2
        self.output = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, segment_ids=None, labels=None):
        # input_ids, input_mask, segment_ids, labels=labels_ids
        sequence_output, cls_output = self.roberta(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
        logits = self.output(cls_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits








