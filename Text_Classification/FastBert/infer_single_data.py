"""
@file  : infer_single_data.py

@author: xiaolu

@time  : 2020-06-03

"""

import argparse
import json
import time
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from model.model_fastbert import FastBertModel, BertConfig
from data_utils.dataset_preparing import PrepareDataset, TextCollate
from utils import load_json_config, init_bert_adam_optimizer, load_saved_model, save_model, eval_pr
import data_utils.tokenization as tokenization

# 随机数固定，RE-PRODUCIBLE
seed = 9999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

debug_break = False


def main(args):
    # 1. 加载配置文件
    config = load_json_config(args.model_config_file)

    # 2. 加载模型
    bert_config = BertConfig.from_json_file(config.get("bert_config_path"))
    model = FastBertModel(bert_config, config)
    load_saved_model(model, args.save_model_path)
    model = model.to(device)
    print('Initialize model Done'.center(60, '*'))

    max_seq_len = 60
    labels = []
    texts = []
    inference_speed = 0.5
    with open('./data/tcl/test.tsv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            label, text = line.split('	')
            labels.append(int(label))
            texts.append(text)
    sum_num = len(labels)

    correct_num = 0
    result = []
    for l, t in zip(labels, texts):
        start_time = time.time()
        # 3. 数据集的准备
        vocab_file = config.get("vocab_file")
        do_lower_case = True
        tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        tokens = tokenizer.tokenize(t)
        tokens = tokens[:(max_seq_len - 1)]
        tokens = ["[CLS]"] + tokens
        tokens = tokenizer.convert_tokens_to_ids(tokens)
        # return {"text": t, "tokens": tokens, "label": label}
        # 4. 开始infer
        segment_ids = [0] * len(tokens)
        attn_masks = [1] * len(tokens)
        tokens = torch.LongTensor([tokens])
        segment_ids = torch.LongTensor([segment_ids])
        attn_masks = torch.LongTensor([attn_masks])
        l = torch.LongTensor([l])
        # print(tokens.size())
        # print(segment_ids.size())
        # print(attn_masks.size())
        # print(l.size())
        with torch.no_grad():
            probs, layer_idxes, uncertain_infos = model(tokens, token_type_ids=segment_ids, attention_mask=attn_masks,
                                                        inference=True, inference_speed=inference_speed)
        _, top_index = probs.topk(1)
        spend_time = time.time() - start_time

        if top_index.view(-1) == l:
            correct_num += 1
        print(l[0].numpy())
        print(top_index.view(-1)[0].numpy())
        exit()


        s = str(l[0]) + '  ' + str(top_index.view(-1)[0]) + '  ' + str(spend_time) + '  ' + t
        result.append(s)
    print('正确率:{}'.format(correct_num / sum_num))
    with open('result.txt', 'w') as f:
        f.write('\n'.join(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Textclassification training script arguments.")
    parser.add_argument("--model_config_file", dest="model_config_file", action="store",
                        help="The path of configuration json file.")

    parser.add_argument("--save_model_path", dest="save_model_path", action="store",
                        help="The path of trained checkpoint model.")

    parser.add_argument("--infer_data", dest="infer_data", action="store", help="")
    parser.add_argument("--dump_info_file", dest="dump_info_file", action="store", help="")

    parser.add_argument("--inference_speed", dest="inference_speed", action="store",
                        type=float, default=1.0, help="")

    # -1 for NO GPU
    parser.add_argument("--gpu_ids", dest="gpu_ids", action="store", default="0",
                        help="Device ids of used gpus, split by ',' , IF -1 then no gpu")

    parser.add_argument("--data_load_num_workers", dest="data_load_num_workers", action="store", type=int, default=1,
                        help="")
    parser.add_argument("--debug_break", dest="debug_break", action="store", type=int, default=0,
                        help="Running debug_break, 0 or 1.")

    parsed_args = parser.parse_args()
    # debug_break = (parsed_args.debug_break == 1)
    main(parsed_args)
