# -*- coding: utf-8 -*-
# @Time    : 2020/9/10 9:15
# @Author  : xiaolu
# @FileName: bert_corrector.py
# @Software: PyCharm
import torch
from torch import nn
from transformers import BertTokenizer, BertConfig
from transformers import BertModel
from transformers import BertForPreTraining
from pdb import set_trace


if __name__ == "__main__":
    # 预处理数据
    tokenizer = BertTokenizer.from_pretrained('./retrain_bert/vocab.txt')

    # 对句子进行编码
    input = "产拳制度不够明悉，政腐作为自然资元掌握着"
    input_ids = tokenizer.encode(input)
    input_ids = torch.LongTensor([input_ids])

    # 加载模型
    config = BertConfig.from_pretrained('./retrain_bert/config.json')

    bert_model = BertForPreTraining.from_pretrained('./retrain_bert/pytorch_model_epoch40000.bin', config=config)

    sequence_output, cls_output = bert_model(input_ids)
    # print(sequence_output.size())   # torch.Size([1, 22, 21128])

    sequence_output = sequence_output[:, 1:-1, :]
    sequence_output = sequence_output.squeeze(0)

    values, indices = torch.max(sequence_output, dim=1)
    indices = indices.numpy().tolist()

    output = tokenizer.decode(indices)
    output = output.replace(' ', '')
    print(output)   # 产权制度不够明熟，政权作为自然资源掌握着









