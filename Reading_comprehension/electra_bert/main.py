# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 15:07
# @Author  : xiaolu
# @FileName: main.py
# @Software: PyCharm
from data_process import read_examples, convert_examples_to_features
from transformers import BertTokenizer, BertConfig, BertModel
from model import BertSupportNet
from utils import convert_to_tokens
from data_helper import DataHelper
import numpy as np
from os.path import join
from data_process import InputFeatures, Example
import torch
from config import set_config
import argparse
import json
import gzip
import pickle
from tqdm import tqdm
from transformers import BertTokenizer


infile = "../input/data.json"
outfile = "../result/result.json"


@torch.no_grad()
def predict(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):
    model.eval()
    answer_dict = {}
    sp_dict = {}
    dataloader.refresh()
    total_test_loss = [0] * 5

    for batch in tqdm(dataloader):

        batch['context_mask'] = batch['context_mask'].float()
        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)
        # loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)

        # for i, l in enumerate(loss_list):
        #     if not isinstance(l, int):
        #         total_test_loss[i] += l.item()

        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                         start_position.data.cpu().numpy().tolist(),
                                         end_position.data.cpu().numpy().tolist(),
                                         np.argmax(type_logits.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = batch['ids'][i]

            cur_sp_logit_pred = []  # for sp logit output
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if need_sp_logit_file:
                    temp_title, temp_id = example_dict[cur_id].sent_names[j]
                    cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                if predict_support_np[i, j] > 0.5:
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            sp_dict.update({cur_id: cur_sp_pred})

    new_answer_dict = {}
    for key, value in answer_dict.items():
        new_answer_dict[key] = value.replace(" ", "")
    prediction = {'answer': new_answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w', encoding='utf8') as f:
        json.dump(prediction, f, indent=4, ensure_ascii=False)


def main():
    tokenizer = BertTokenizer.from_pretrained('./albert_pretrain/vocab.txt')
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    # 1. 加载数据
    examples = read_examples(full_file=infile)  # 输入数据的文件夹

    # 2. 数据预处理
    with gzip.open('./dev_example.pkl.gz', 'wb') as fout:
        pickle.dump(examples, fout)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length=512, max_query_length=50)
    with gzip.open('./dev_feature.pkl.gz', 'wb') as fout:
        pickle.dump(features, fout)

    args = set_config()
    helper = DataHelper(gz=True, config=args)
    dev_example_dict = helper.dev_example_dict
    dev_feature_dict = helper.dev_feature_dict
    eval_dataset = helper.dev_loader

    roberta_config = BertConfig.from_pretrained('./albert_pretrain/bert_config.json')
    encoder = BertModel.from_pretrained(args.bert_model, config=roberta_config)
    model = BertSupportNet(config=args, encoder=encoder)

    model.load_state_dict(torch.load('./output/checkpoints/train_v1/ckpt_seed_44_epoch_20_99999.pth', map_location={'cuda:6': 'cuda:0'}))

    model.to(device)

    predict(model, eval_dataset, dev_example_dict, dev_feature_dict, outfile)


if __name__ == '__main__':
    main()
