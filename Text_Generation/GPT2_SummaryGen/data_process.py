# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 11:42
# @Author  : xiaolu
# @FileName: data_process.py
# @Software: PyCharm
from tqdm import tqdm
import argparse
import json
from transformers import BertTokenizer
from config import Config


def preprocess_raw_data(tokenizer, n_ctx):
    """
    对原始语料进行处理，将原始语料转换为用于train的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    :param args:
    :param tokenizer:
    :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
    :return:
    """
    with open(Config.train_tokenized_path, 'w', encoding='utf8') as f:
        with open(Config.train_raw_path, 'r', encoding='utf8') as file:
            for line in tqdm(file.readlines()):
                try:
                    file_line = json.loads(line)
                except:
                    print('line:', line)
                else:
                    dialogue_ids = [tokenizer.cls_token_id]
                    dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in file_line['article']])
                    dialogue_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
                    dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in file_line['summarization']])
                    dialogue_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
                    # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
                    dialogue_ids = dialogue_ids[:n_ctx]
                    for dialogue_id in dialogue_ids:
                        f.write(str(dialogue_id) + ' ')
                    f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(Config.gpt2_vocab)
    n_ctx = 1024
    preprocess_raw_data(tokenizer, n_ctx)


