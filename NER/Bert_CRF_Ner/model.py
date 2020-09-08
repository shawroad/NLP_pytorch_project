# -*- coding: utf-8 -*-
# @Time    : 2020/9/4 14:12
# @Author  : xiaolu
# @FileName: model.py
# @Software: PyCharm
import torch.nn as nn
from crf import CRF
from transformers import BertModel, BertConfig


class BertCrfForNer(nn.Module):
    def __init__(self, num_labels):
        super(BertCrfForNer, self).__init__()
        # bert模型
        self.config = BertConfig.from_pretrained('./bert_pretrain/bert_config.json')
        self.bert = BertModel.from_pretrained('./bert_pretrain/pytorch_model.bin', config=self.config)

        # 每个token进行分类
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # 送入CRF进行预测
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]   # B, L, H

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1*loss,) + outputs
        return outputs # (loss), scores