"""
@file   : inference.py.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-01-05
"""
import torch
from config import set_args
from model_cls import Model
from transformers.models.bert import BertTokenizer
from data_helper import clean_text, pad_to_maxlen


def convert_text_ids(text):
    inputs = tokenizer.encode_plus(
        text=text,
        text_pair=None,
        add_special_tokens=True,
        return_token_type_ids=True
    )
    max_len = 128
    input_ids = pad_to_maxlen(inputs['input_ids'], max_len=max_len)
    attention_mask = pad_to_maxlen(inputs['attention_mask'], max_len=max_len)
    token_type_ids = pad_to_maxlen(inputs["token_type_ids"], max_len=max_len)

    all_input_ids = torch.tensor([input_ids], dtype=torch.long)
    all_input_mask = torch.tensor([attention_mask], dtype=torch.long)
    all_segment_ids = torch.tensor([token_type_ids], dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids


if __name__ == '__main__':
    args = set_args()
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    model = Model()
    model.load_state_dict(torch.load('./outputs/base_model_epoch_{}.bin'.format(0)))
    model.eval()

    # 单条样本推理
    text1 = '酷狗音乐 [小黄人高兴]10月2日 #酷狗蘑菇国风动漫音乐玩唱会#重磅来袭 购票戳 [憧憬]@阿杰7'
    text1 = clean_text(text1)

    input_ids, input_mask, segment_ids = convert_text_ids(text1)
    if torch.cuda.is_available():
        input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=input_mask, segment_ids=segment_ids)
        # print(logits.size())   # torch.Size([1, 21])
        prob = logits.cpu().detach().numpy().tolist()[0]
        cls = prob.index(max(prob))
        print(cls)   # 然后转为id2label 即可
