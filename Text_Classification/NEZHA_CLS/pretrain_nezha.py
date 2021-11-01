"""
@file   : pretrain_nezha.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-09-30
"""
import torch
import json
import random
import numpy as np
from itertools import chain
from NEZHA.modeling_nezha import NeZhaForMaskedLM
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers.models.bert import BertTokenizer
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter


def load_data(path):
    corpus = []
    with open(path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            sample = json.loads(line.strip())
            text = sample['text']
            corpus.append(text)
    return corpus


class MLM_Data(Dataset):
    # 传入句子对列表
    def __init__(self, text, max_len):
        super(Dataset, self).__init__()
        self.data = text
        self.max_len = max_len

        self.spNum = len(tokenizer.all_special_tokens)
        self.tkNum = tokenizer.vocab_size

    def __len__(self):
        return len(self.data)

    def random_mask(self, text_ids):
        input_ids, output_ids = [], []
        rands = np.random.random(len(text_ids))
        idx = 0
        mask_p = 0.5  # 原始是0.15，加大mask_p就会加大预训练难度
        while idx < len(rands):
            if rands[idx] < mask_p:
                # 需要mask
                # n-gram 动态mask策略
                ngram = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])  # 若要mask，进行x_gram mask的概率
                if ngram == 3 and len(rands) < 7:  # 太大的gram不要应用于过短文本
                    ngram = 2
                if ngram == 2 and len(rands) < 4:
                    ngram = 1

                L = idx + 1
                R = idx + ngram  # 最终需要mask的右边界（开）
                while L < R and L < len(rands):
                    rands[L] = np.random.random() * 0.15  # 强制mask
                    L += 1
                idx = R
                if idx < len(rands):
                    rands[idx] = 1  # 禁止mask片段的下一个token被mask，防止一大片连续mask
            idx += 1

        for r, i in zip(rands, text_ids):
            if r < mask_p * 0.8:
                input_ids.append(tokenizer.mask_token_id)
                output_ids.append(i)  # mask预测自己
            elif r < mask_p * 0.9:
                input_ids.append(i)
                output_ids.append(i)  # 自己预测自己
            elif r < mask_p:
                input_ids.append(np.random.randint(self.spNum, self.tkNum))
                output_ids.append(i)  # 随机的一个词预测自己，随机词不会从特殊符号中选取，有小概率抽到自己
            else:
                input_ids.append(i)
                output_ids.append(-100)  # 保持原样不预测

        return input_ids, output_ids

    # 耗时操作在此进行，可用上多进程
    def __getitem__(self, item):
        text1 = self.data[item]  # 预处理，mask等操作
        text1 = truncate(text1, self.max_len)
        text1_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1))

        text1_ids, out1_ids = self.random_mask(text1_ids)  # 添加mask预测
        input_ids = [tokenizer.cls_token_id] + text1_ids + [tokenizer.sep_token_id]  # 拼接
        token_type_ids = [0] * (len(text1_ids) + 2)
        labels = [-100] + out1_ids + [-100]
        assert len(input_ids) == len(token_type_ids) == len(labels)
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'labels': labels}

    @classmethod
    def collate(cls, batch):
        input_ids = [i['input_ids'] for i in batch]
        token_type_ids = [i['token_type_ids'] for i in batch]
        labels = [i['labels'] for i in batch]
        input_ids = paddingList(input_ids, 0, returnTensor=True)
        token_type_ids = paddingList(token_type_ids, 0, returnTensor=True)
        labels = paddingList(labels, -100, returnTensor=True)
        attention_mask = (input_ids != 0)
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                'labels': labels}


def collate(batch):
    input_ids = [i['input_ids'] for i in batch]
    token_type_ids = [i['token_type_ids'] for i in batch]
    labels = [i['labels'] for i in batch]
    input_ids = paddingList(input_ids, 0, returnTensor=True)
    token_type_ids = paddingList(token_type_ids, 0, returnTensor=True)
    labels = paddingList(labels, -100, returnTensor=True)
    attention_mask = (input_ids != 0)
    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                'labels': labels}


def paddingList(ls: list, val, returnTensor=False):
    ls = ls[:]  # 不要改变了原list尺寸
    maxLen = max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i] = ls[i] + [val] * (maxLen - len(ls[i]))

    return torch.tensor(ls) if returnTensor else ls


def truncate(a: list, maxLen):
    maxLen -= 3  # 空留给cls sep sep
    assert maxLen >= 0
    # 一共就a超长与否，b超长与否，组合的四种情况
    if len(a) > maxLen:  # 需要截断
        # 尾截断
        # a=a[:maxLen]
        # 首截断
        # a = a[maxLen-len(a):]
        # 首尾截断
        outlen = (len(a) - maxLen)
        headid = int(outlen / 2)
        a = a[headid: headid - outlen]
    return a


if __name__ == "__main__":
    text = []
    train_data = load_data('./data/train.json')
    dev_data = load_data('./data/test.json')

    tokenizer = BertTokenizer.from_pretrained('./nezha_pretrain')
    max_len = 128
    batch_size = 8
    train_MLM_data = MLM_Data(train_data, max_len)

    training_args = TrainingArguments(
        output_dir='/Users/xiaolu10/Desktop/Project/NEZHA_CLS/retrain_nezha_pretrain',  # 此处必须是绝对路径
        overwrite_output_dir=True,
        num_train_epochs=1000,
        per_device_train_batch_size=batch_size,
        save_steps=(len(train_data) // batch_size) * 10000,  # 每10个epoch save一次
        save_total_limit=3,
        # logging_steps=len(dl),  # 每个epoch log一次
        logging_steps=2,  # 每个epoch log一次
        seed=2021,
        learning_rate=5e-5,
        weight_decay=0.01,
        prediction_loss_only=True,
        warmup_steps=int(450000 * 150 / batch_size * 0.03)
    )
    model = NeZhaForMaskedLM.from_pretrained("./nezha_pretrain")
    if torch.cuda.is_available():
        model.cuda()
    model.resize_token_embeddings(len(tokenizer))
    # 只训练word_embedding。能缩短两倍的训练时间
    for name, p in model.named_parameters():
        if name != 'bert.embeddings.word_embeddings.weight':
            p.requires_grad = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=MLM_Data(text=train_data, max_len=max_len),
        data_collator=collate
    )

    trainer.train()
    trainer.save_model('./nezha_model')


