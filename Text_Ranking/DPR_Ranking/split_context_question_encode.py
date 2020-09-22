# -*- coding: utf-8 -*-
# @Time    : 2020/9/22 11:05
# @Author  : xiaolu
# @FileName: split_context_question_encode.py
# @Software: PyCharm
import torch
from model import Model
import collections
from transformers import BertModel, BertConfig
from transformers import BertTokenizer


if __name__ == '__main__':
    model1 = Model()
    model1.load_state_dict(torch.load('./output/q_c_encode_epoch0.bin', map_location='cpu'))

    # 直接看权重的名字和权重  推荐一般用这个
    ques_dict = {}
    context_dict = {}
    for name, parameters in model1.named_parameters():
        if name.split('.')[0] == 'roberta_ques':
            temp = '.'.join(name.split('.')[1:])
            ques_dict[temp] = parameters
        if name.split('.')[0] == 'roberta_context':
            temp = '.'.join(name.split('.')[1:])
            context_dict[temp] = parameters

    # 保存问题模型
    config = BertConfig.from_pretrained('./roberta_pretrain/bert_config.json')
    ques_model = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=config)
    ques_model.load_state_dict(ques_dict)
    question_model_file = "./output/question_encode_model.bin"
    torch.save(ques_model.state_dict(), question_model_file)

    # 保存文章模型
    config = BertConfig.from_pretrained('./roberta_pretrain/bert_config.json')
    context_model = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=config)
    context_model.load_state_dict(context_dict)
    context_model_file = "./output/context_encode_model.bin"
    torch.save(context_model.state_dict(), context_model_file)

    # # 把上面隐藏了 小测一下
    # tokenizer = BertTokenizer.from_pretrained('./roberta_pretrain/vocab.txt')
    # text = '我在马路边捡到一分钱，把他交给警察叔叔手里面'
    # input_ids = torch.LongTensor([tokenizer.encode(text)])
    #
    # # 保存问题模型
    # config = BertConfig.from_pretrained('./roberta_pretrain/bert_config.json')
    # ques_model = BertModel.from_pretrained('./output/question_encode_model.bin', config=config)
    # q_seq_encode, q_cls_encode = ques_model(input_ids)
    # print(q_cls_encode)
    #
    # # 保存文章模型
    # config = BertConfig.from_pretrained('./roberta_pretrain/bert_config.json')
    # context_model = BertModel.from_pretrained('./output/context_encode_model.bin', config=config)
    # c_seq_encode, c_cls_encode = context_model(input_ids)
    # print(c_cls_encode)


