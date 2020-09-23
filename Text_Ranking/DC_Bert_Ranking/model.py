# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 9:33
# @Author  : xiaolu
# @FileName: model.py
# @Software: PyCharm
import torch
from torch import nn
from transformers import BertModel, BertConfig
from transformers import BertLayer
from torch.nn import CrossEntropyLoss
from pdb import set_trace


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./roberta_pretrain/bert_config.json')
        self.ques_encoder = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.config)
        self.context_encoder = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.config)

        self.basicblocks = nn.ModuleList()
        self.n_layers = 3
        trans_heads = 8
        trans_drop = 0.1
        bert_config = BertConfig(hidden_size=self.config.hidden_size, num_attention_heads=trans_heads, attention_probs_dropout_prob=trans_drop)

        for layer in range(self.n_layers):
            self.basicblocks.append(BertLayer(bert_config))

        self.num_labels = 2
        self.output = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, ques_input_ids=None, ques_input_mask=None, ques_segment_ids=None,
                context_input_ids=None, context_input_mask=None, context_segment_ids=None, labels=None):
        ques_seq_output, ques_cls_output = self.ques_encoder(input_ids=ques_input_ids,
                                                             token_type_ids=ques_segment_ids,
                                                             attention_mask=ques_input_mask)

        # print(ques_seq_output.size())    # torch.Size([2, 61, 768])
        context_seq_output, context_cls_output = self.ques_encoder(input_ids=context_input_ids,
                                                                   token_type_ids=context_segment_ids,
                                                                   attention_mask=context_input_mask)
        # print(context_seq_output.size())   # torch.Size([2, 441, 768])

        concat_encode = torch.cat([ques_seq_output, context_seq_output], dim=1)   # torch.Size([2, 502, 768])
        concat_mask = torch.cat([ques_input_mask, context_input_mask], dim=1)

        temp_attention_mask = concat_mask.unsqueeze(1).unsqueeze(2)  # batch_size, 1, 1, 45
        temp_attention_mask = (1.0 - temp_attention_mask) * -10000.0  # 零位置是-10000 1位置是零

        for l in range(self.n_layers):
            concat_encode = self.basicblocks[l](concat_encode, temp_attention_mask)[0]
        # print(concat_encode.size())   # torch.Size([2, 502, 768])
        mask = (1.0 - concat_mask.unsqueeze(-1)) * - 10000.0
        token_prob = (concat_encode + mask).softmax(dim=-2)
        output_state = torch.sum(token_prob * concat_encode, dim=1)
        # print(output.size())   # torch.Size([2, 768])
        logits = self.output(output_state)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits







