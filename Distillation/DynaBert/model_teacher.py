# -*- encoding: utf-8 -*-
'''
@File    :   model_teacher.py
@Time    :   2020/10/19 09:49:29
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
from my_transformers import BertConfig, BertModel
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class Teacher_Model(nn.Module):
    def __init__(self, args):
        super(Teacher_Model, self).__init__()
        self.label_nums = 2
        self.config = BertConfig.from_pretrained(args.model_config)
        self.config.output_hidden_states = True    # 返回每层的输出
        self.config.output_intermediate = False    # 是否返回前馈网络的中间层
        self.bert = BertModel.from_pretrained(args.pretrain_model, config=self.config)
        self.output = nn.Linear(self.config.hidden_size, self.label_nums)

    def forward(self, input_ids=None, attention_mask=None, segment_ids=None, labels=None):
        final_layer, cls_output, layer_13_output = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
        # print(final_layer.size())     # torch.Size([8, 512, 768])
        # print(cls_output.size())     # torch.Size([8, 768])
        # print(len(layer_13_output))   #  13
        logits = self.output(cls_output)    # size: (batch_size, label_nums)
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.label_nums), labels.view(-1))
        #     return loss, logits
        return logits, layer_13_output

