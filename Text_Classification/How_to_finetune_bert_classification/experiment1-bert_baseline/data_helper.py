"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-29$
"""
import torch
import pandas as pd
from tqdm import tqdm
from config import set_args

args = set_args()


def load_data(path, tokenizer):
    df = pd.read_csv(path, sep='\t')
    df.columns = ['label', 'content']
    # print(df.head())

    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    for lab, cont in tqdm(zip(df['label'].tolist(), df['content'].tolist())):
        text = tokenizer.tokenize(cont)
        if len(text) > args.max_seq_length - 2:
            text = text[:args.max_seq_length - 2]
        text = ['[CLS]'] + text + ['[SEP]']
        attention_mask.append([1] * len(text) + [0] * (args.max_seq_length - len(text)))
        token_type_ids.append([0] * args.max_seq_length)
        input_ids.append(tokenizer.convert_tokens_to_ids(text) + [0] * (args.max_seq_length - len(text)))
        labels.append(int(lab))

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, token_type_ids, labels


