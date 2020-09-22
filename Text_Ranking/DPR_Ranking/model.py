# -*- coding: utf-8 -*-
# @Time    : 2020/9/17 9:25
# @Author  : xiaolu
# @FileName: model.py
# @Software: PyCharm
from torch import nn
from transformers import BertConfig, BertModel


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./roberta_pretrain/bert_config.json')
        self.roberta_ques = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.config)
        self.roberta_context = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.config)

    def forward(self, question_input_ids=None, question_input_mask=None, question_segment_ids=None,
                context_input_ids=None, context_input_mask=None, context_segment_ids=None):

        # 对问题编码
        '''
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        '''
        q_sequence_output, q_cls_output = self.roberta_ques(input_ids=question_input_ids,
                                                            attention_mask=question_input_mask,
                                                            token_type_ids=question_segment_ids)
        c_sequence_output, c_cls_output = self.roberta_context(input_ids=context_input_ids,
                                                               attention_mask=context_input_mask,
                                                               token_type_ids=context_segment_ids)
        return q_cls_output, c_cls_output


