# -*- coding: utf-8 -*-
# @Time    : 2020/9/5 17:58
# @Author  : xiaolu
# @FileName: inference.py
# @Software: PyCharm
import os
import json
import torch
from model import BertCrfForNer
from transformers import BertTokenizer
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from data_process import CnerProcessor as Processor
from data_process import collate_fn, get_entities


def json_to_text(file_path,data):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


def convert_examples_to_features(x_token,
                                 tokenizer,
                                 max_seq_length=512,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True, ):
    if len(x_token) > max_seq_length-2:
        x_token = x_token[: (max_seq_length - 2)]

    input_len = len(x_token)
    x_token += [sep_token]
    segment_ids = [sequence_a_segment_id] * len(x_token)
    x_token = [cls_token] + x_token
    segment_ids = [cls_token_segment_id] + segment_ids
    input_ids = tokenizer.convert_tokens_to_ids(x_token)

    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    padding_length = max_seq_length - len(input_ids)

    input_ids += [pad_token] * padding_length
    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
    segment_ids += [pad_token_segment_id] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    print("*** Example ***")
    print("tokens: {}".format(" ".join([str(x) for x in x_token])))
    print("input_ids: {}".format(" ".join([str(x) for x in input_ids])))
    print("input_mask: {}".format(" ".join([str(x) for x in input_mask])))
    print("segment_ids: {}".format(" ".join([str(x) for x in segment_ids])))

    return x_token, input_ids, input_mask, segment_ids, input_len


def predict(tokens, input_ids, input_mask, segment_ids, input_len):
    with torch.no_grad():
        input_ids = torch.LongTensor([input_ids])
        input_mask = torch.LongTensor([input_mask])
        segment_ids = torch.LongTensor([segment_ids])
        inputs = {'input_ids': input_ids, 'attention_mask': input_mask}
        outputs = model(**inputs)
        logits = outputs[0]
        # print(logits.size())   # torch.Size([1, 512, 23])
        tags = model.crf.decode(logits, inputs['attention_mask'])
        print(tags.size())   # torch.Size([1, 1, 512])
        tags = tags.squeeze(0).cpu().numpy().tolist()

    preds = tags[0][1: -1]   # 取出CLS SEP
    label_entities = get_entities(preds, id2label)
    json_d = {}
    json_d['tag_seq'] = ' '.join([id2label[x] for x in preds])
    json_d['entities'] = label_entities
    print(tokens[1:-1])
    print(json_d['tag_seq'].split(' ')[:input_len])
    print(len(tokens[1:-1]))
    print(len(json_d['tag_seq'].split(' ')[:input_len]))


if __name__ == '__main__':
    processor = Processor()
    label_list = processor.get_labels()
    # 将标签进行id映射
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    num_labels = len(label_list)
    # s = '常建良，男，1963年出生，工科学士，高级工程师，北京物资学院客座副教授。'
    s = ['1', '9', '6', '6', '年', '出', '生', '，', '汉', '族', '，', '中', '共', '党', '员', '，', '本', '科', '学',
         '历', '，', '工', '程', '师', '、', '美', '国', '项', '目', '管', '理', '协', '会', '注', '册', '会', '员', '（',
         'P', 'M', 'I', 'M', 'e', 'm', 'b', 'e', 'r', '）', '、', '注', '册', '项', '目', '管', '理', '专', '家', '（',
         'P', 'M', 'P', '）', '、', '项', '目', '经', '理', '。']
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain/vocab.txt')
    tokens, input_ids, input_mask, segment_ids, input_len = convert_examples_to_features(s, tokenizer)

    model = BertCrfForNer(num_labels)
    model.load_state_dict(torch.load('./save_model/ckpt_epoch_2.bin', map_location='cpu'))

    predict(tokens, input_ids, input_mask, segment_ids, input_len)


