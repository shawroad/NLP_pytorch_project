# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:30
# @Author  : xiaolu
# @FileName: inference.py
# @Software: PyCharm
import json
import torch
from transformers import BertTokenizer
from model import Model
from config import set_args


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(question, context_list, max_seq_length, tokenizer):
    features = []
    for context in context_list:
        temp = tokenizer.encode_plus(question, context, max_length=max_seq_length, pad_to_max_length=True)
        # temp: {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        features.append(temp)
    return features


def evaluate(features):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    # print(input_ids.size())   # torch.Size([7, 512])
    # print(input_mask.size())   # torch.Size([7, 512])
    # print(segment_ids.size())    # torch.Size([7, 512])
    predictions = []
    with torch.no_grad():
        # forward(self, input_ids=None, attention_mask=None, segment_ids=None, labels=None)
        logits = model(input_ids=input_ids, attention_mask=input_mask, segment_ids=segment_ids)
        # 预测为正样本的概率
        pos_pred = logits[:, 1].tolist()
        index = sorted(enumerate(pos_pred), key=lambda x: -x[1])

        # 按相关度的大小进行文档排序
        result = []
        for i, _ in index:
            result.append(i)
    return result


if __name__ == "__main__":
    args = set_args()
    device = torch.device('cuda: {}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    # # 1. 加载模型
    model = Model()
    model.load_state_dict(torch.load('./save_model/best_pytorch_model.bin', map_location='cpu'))
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_file)

    # 加载需推理的数据
    data = json.load(open('./data/test.json', 'r', encoding='utf8'))
    max_seq_length = 512

    for item in data:
        question = item['question']
        context_list = item['context']
        features = convert_examples_to_features(question, context_list, max_seq_length, tokenizer)
        # print(features)
        rank_index = evaluate(features)
        result = []
        for i in rank_index:
            res = context_list[i]
            result.append(res)
        # temp = {'question': question,
        #         'rank_context': result}
        print('问题:', question)
        for i, r in enumerate(result):
            print("第{}篇: {}".format(i+1, r))
        print('*'*300)


