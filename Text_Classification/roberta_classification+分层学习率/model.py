# -*- coding: utf-8 -*-
# @Time    : 2020/8/26 17:28
# @Author  : xiaolu
# @FileName: model.py
# @Software: PyCharm
import torch
from torch import nn
from transformers import BertModel, BertConfig
from torch.nn import CrossEntropyLoss, MSELoss
from pdb import set_trace


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_labels = 3
        self.config = BertConfig.from_pretrained('./roberta_base/bert_config.json')
        self.roberta = BertModel.from_pretrained('./roberta_base/pytorch_model.bin', config=self.config)

        self.concat_squeeze = nn.Linear(2 * self.config.hidden_size, self.config.hidden_size)

        self.output = nn.Linear(self.config.hidden_size, self.num_labels)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        '''
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param labels:
        :return:
        '''

        last_layer_output, pooled_output, all_layers_output = self.roberta(input_ids=input_ids,
                                                                           attention_mask=attention_mask,
                                                                           token_type_ids=token_type_ids)
        # last_output, last_2_output = all_layers_output[-1], all_layers_output[-2]
        # concat_output = torch.cat((last_output, last_2_output), dim=-1)

        # x = self.concat_squeeze(concat_output)
        logits = self.output(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits









