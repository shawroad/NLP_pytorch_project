"""
@file   : inference.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-04-07
"""
import torch
import json
import numpy as np
from tqdm import tqdm
from model import GlobalPointer
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertModel
from data_helper import DataMaker, MyDataset


def load_data(data_path, data_type="test"):
    if data_type == "test":
        datas = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                line = json.loads(line)
                datas.append(line)
        return datas
    else:
        return json.load(open(data_path, encoding="utf-8"))


def data_generator(test_data_path):
    test_data = load_data(test_data_path, "test")
    all_data = test_data
    max_tok_num = 0
    for sample in all_data:
        tokens = tokenizer.tokenize(sample["text"])
        max_tok_num = max(max_tok_num, len(tokens))

    max_seq_len = 256
    data_maker = DataMaker(tokenizer)
    test_dataloader = DataLoader(MyDataset(test_data), batch_size=16, shuffle=False,
                                 collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id, data_type="test"))

    return test_dataloader


def decode_ent(text, pred_matrix, tokenizer, threshold=0):
    # print(text)
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    id2ent = {id: ent for ent, id in ent2id.items()}
    pred_matrix = pred_matrix.cpu().numpy()
    ent_list = {}
    for ent_type_id, token_start_index, token_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = id2ent[ent_type_id]
        ent_char_span = [token2char_span_mapping[token_start_index][0], token2char_span_mapping[token_end_index][1]]
        ent_text = text[ent_char_span[0]:ent_char_span[1]]

        ent_type_dict = ent_list.get(ent_type, {})
        ent_text_list = ent_type_dict.get(ent_text, [])
        ent_text_list.append(ent_char_span)
        ent_type_dict.update({ent_text: ent_text_list})
        ent_list.update({ent_type: ent_type_dict})
    return ent_list


def predict(dataloader, model):
    predict_res = []
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, _ = batch_data
        if torch.cuda.is_available():
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()

        with torch.no_grad():
            batch_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)

        for ind in range(len(batch_samples)):
            gold_sample = batch_samples[ind]
            text = gold_sample["text"]
            text_id = gold_sample["id"]
            pred_matrix = batch_logits[ind]
            labels = decode_ent(text, pred_matrix, tokenizer)
            predict_res.append({"id": text_id, "text": text, "label": labels})
    return predict_res


if __name__ == '__main__':
    ent2id = load_data('./data/ent2id.json', "ent2id")
    ent_type_size = len(ent2id)
    tokenizer = BertTokenizerFast.from_pretrained('./roberta_pretrain', add_special_tokens=True, do_lower_case=False)

    model = GlobalPointer(ent_type_size)
    model.load_state_dict(torch.load('./output/best_model.bin'))
    if torch.cuda.is_available():
        model.cuda()

    # 准备数据
    # test_data_path = './data/test.json'
    test_data_path = './ad_data.json'
    test_dataloader = data_generator(test_data_path)

    predict_res = predict(test_dataloader, model)

    with open('predict_result.json', "w", encoding="utf-8") as f:
        for item in predict_res:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
