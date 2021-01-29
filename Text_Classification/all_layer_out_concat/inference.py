"""
# -*- coding: utf-8 -*-
# @File    : inference.py
# @Time    : 2021/1/26 7:06 下午
# @Author  : xiaolu
# @Email   : luxiaonlp@163.com
# @Software: PyCharm
"""
import torch
import json
from transformers import BertTokenizer
import torch.nn.functional as F
from model import Model

def convert_id(question, context):
    input_id = []
    input_mask = []
    segment_id = []

    # 1. 加入CLS
    input_id.append(tokenizer.cls_token_id)

    # 2. 将问题编码 然后加入
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question))
    input_id.extend(ids)
    segment_id.extend([0] * len(input_id))

    # 3. 加入sep
    input_id.append(tokenizer.sep_token_id)

    # 4. 加入文章
    ids_ = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
    input_id.extend(ids_)
    segment_id.extend([1] * (len(input_id) - len(segment_id)))

    # 5. 进行padding
    if len(input_id) >= 512:
        input_id = input_id[:511]
        segment_id = segment_id[:511]
        # 长度大于等于512 截成511 加入sep
        input_id.append(tokenizer.sep_token_id)
        segment_id.append(1)
        input_mask.extend([1] * 512)
    else:
        # 加入sep
        input_id.append(tokenizer.sep_token_id)
        input_mask.extend([1] * len(input_id))
        segment_id.append(1)

    # 填充
    input_id.extend([0] * (512 - len(input_id)))
    input_mask.extend([0] * (512 - len(input_mask)))
    segment_id.extend([0] * (512 - len(segment_id)))

    assert len(input_id) == len(input_mask) and len(input_id) == len(segment_id)
    return input_id, input_mask, segment_id


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('./roberta_pretrain/vocab.txt')
    model = Model()
    model.load_state_dict(torch.load('./save_teacher_model/best_pytorch_model.bin', map_location='cpu'))
    print("模型加载成功...")
    model.eval()

    dev_data = json.load(open('./data/mini_data/dev_mini.json'))
    result = []
    for item in dev_data:
        question = item['question']
        doc = item['related_doc']
        answer = item['answer']
        score_list = []
        for d in doc:
            text = d['body']
            keywords = ','.join(d['keywords'])
            try:
                title = item['title']
            except:
                title = ''
            context = keywords + title + text
            es_score = d['score']

            # 将文章和问题拼接，送入模型计算得分
            input_id, input_mask, segment_id = convert_id(question, context)

            input_id = torch.tensor([input_id], dtype=torch.long)
            input_mask = torch.tensor([input_mask], dtype=torch.long)
            segment_id = torch.tensor([segment_id], dtype=torch.long)

            with torch.no_grad():
                output = model(input_ids=input_id, attention_mask=input_mask, segment_ids=segment_id)
                output = F.softmax(output)
                score = output[0][1].numpy()   # 模型预测得分
                es_score = es_score / 1000   # es得分
                temp = [score, text]
                score_list.append(temp)

        score_list.sort(key=lambda x: x[0], reverse=True)
        related_doc = [{'body': _[1]} for _ in score_list]

        result.append({'question': question, 'answer': answer, 'related_doc': related_doc})

    # 写入
    json.dump(result, open('./data/mini_data/result.json'), ensure_ascii=False)








