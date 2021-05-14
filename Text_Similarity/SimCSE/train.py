"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-06
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, get_linear_schedule_with_warmup, AdamW
from config import set_args
import torch.nn.functional as F
from model import Model
from torch.utils.tensorboard import SummaryWriter
from utils import compute_corrcoef, l2_normalize


def load_data(path):
    # 加载数据
    D = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            text1, text2, label = line.strip().split('\t')
            D.append((text1, text2, float(label)))
    return D


def split_data(dat):
    # 切分数据
    a_texts, b_texts, labels = [],[],[],
    for d in tqdm(dat):
        a_texts.append(d[0])
        b_texts.append(d[1])
        labels.append(d[2])
    return a_texts, b_texts, labels


class SimCSE_DataSet(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.data_set = []
        for d in self.data:
            input_ids, attention_mask = self.convert_feature(d)
            self.data_set.append({'input_ids': input_ids, 'attention_mask': attention_mask})

    def convert_feature(self, sample):
        # 将文本转为id序列
        input_ids = self.tokenizer.encode(sample)
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def collate_func(batch_data):
    '''
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    :param batch_data: batch数据
    :return:
    '''
    batch_size = len(batch_data)

    # 如果batch_size为0，则返回一个空字典
    if batch_size == 0:
        return {}

    input_ids_list, attention_mask_list = [], []
    for instance in batch_data:
        input_ids_temp = instance['input_ids']
        attention_mask_temp = instance['attention_mask']

        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))

    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "attention_mask_ids": pad_sequence(attention_mask_list, batch_first=True, padding_value=0)}


def simcse_loss(y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = torch.arange(0, y_pred.size(0))  # [b]

    idxs_1 = idxs[None, :]  # [1,b]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]  # [b,1]    4 + 1 = 5   7 + 1 - 1 * 2 = 6
    y_true = idxs_1 == idxs_2

    if torch.cuda.is_available():
        y_true = torch.tensor(y_true, dtype=torch.float).cuda()
    else:
        y_true = torch.tensor(y_true, dtype=torch.float)

    # 计算相似度
    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = torch.matmul(y_pred, y_pred.transpose(0, 1))  # [b,d] * [b.d] -> [b,1]

    if torch.cuda.is_available():
        similarities = similarities - torch.eye(y_pred.size(0)).cuda() * 1e12
    else:
        similarities = similarities - torch.eye(y_pred.size(0)) * 1e12

    similarities = similarities * 20
    loss = loss_func(similarities, y_true)
    return loss


def evaluate():
    model.eval()

    # 语料向量化
    all_vecs = []
    for a_texts, b_texts in all_texts:
        a_data = SimCSE_DataSet(a_texts, tokenizer, max_len)
        a_data_gen = DataLoader(a_data, batch_size=args.train_batch_size, collate_fn=collate_func)

        b_data = SimCSE_DataSet(b_texts, tokenizer, max_len)
        b_data_gen = DataLoader(b_data, batch_size=args.train_batch_size, collate_fn=collate_func)

        all_a_vecs = []
        for eval_batch in tqdm(a_data_gen):
            if torch.cuda.is_available():
                input_ids = eval_batch["input_ids"].cuda()
                attention_mask_ids = eval_batch["attention_mask_ids"].cuda()
            else:
                input_ids = eval_batch["input_ids"]
                attention_mask_ids = eval_batch["attention_mask_ids"]


            with torch.no_grad():
                eval_encodings = model(input_ids=input_ids, attention_mask=attention_mask_ids)
                eval_encodings = eval_encodings.cpu().detach().numpy()
                all_a_vecs.extend(eval_encodings)

        all_b_vecs = []
        for eval_batch in tqdm(b_data_gen):
            if torch.cuda.is_available():
                input_ids = eval_batch["input_ids"].cuda()
                attention_mask_ids = eval_batch["attention_mask_ids"].cuda()
            else:
                input_ids = eval_batch["input_ids"]
                attention_mask_ids = eval_batch["attention_mask_ids"]

            with torch.no_grad():
                eval_encodings = model(input_ids=input_ids, attention_mask=attention_mask_ids)
                eval_encodings = eval_encodings.cpu().detach().numpy()
                all_b_vecs.extend(eval_encodings)

        all_vecs.append((np.array(all_a_vecs), np.array(all_b_vecs)))

    # 标准化，相似度，相关系数
    all_corrcoefs = []
    for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
        a_vecs = l2_normalize(a_vecs)
        b_vecs = l2_normalize(b_vecs)
        sims = (a_vecs * b_vecs).sum(axis=1)
        corrcoef = compute_corrcoef(labels, sims)
        all_corrcoefs.append(corrcoef)

    all_corrcoefs.extend([
        np.average(all_corrcoefs),
        np.average(all_corrcoefs, weights=all_weights)
    ])
    print(all_corrcoefs)
    return all_corrcoefs


if __name__ == '__main__':
    args = set_args()

    # 1. 加载数据
    data_path = './data/LCQMC/'
    # 加载训练集、验证集、测试集
    datasets = {fn: load_data("./data/LCQMC/LCQMC.{}.data".format(fn)) for fn in ['train', 'valid', 'test']}

    all_weights, all_texts, all_labels = [], [], []
    train_texts = []
    for name, data in datasets.items():
        a_texts, b_texts, labels = split_data(data)
        all_weights.append(len(data))   # 存[训练集的个数、测试集的个数、验证集的个数]
        all_texts.append((a_texts, b_texts))
        all_labels.append(labels)
        train_texts.extend(a_texts)
        train_texts.extend(b_texts)

    np.random.shuffle(train_texts)
    train_texts = train_texts[:10000]

    # 2. 构建一个数据加载器
    max_len = 64
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain/vocab.txt')
    train_data = SimCSE_DataSet(train_texts, tokenizer, max_len)

    train_data_loader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=collate_func)

    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    print("总训练步数为:{}".format(total_steps))

    model = Model()

    if torch.cuda.is_available():
        model.cuda()

    # 获取模型所有参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)

    loss_func = nn.BCEWithLogitsLoss()
    tr_loss = 0
    global_step = 0
    logging_loss = 0
    tb_write = SummaryWriter()

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_data_loader):
            input_ids = batch["input_ids"]
            attention_mask_ids = batch["attention_mask_ids"]

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask_ids = attention_mask_ids.cuda()

            outputs = model(input_ids, attention_mask_ids, encoder_type='fist-last-avg')
            loss = simcse_loss(outputs)
            tr_loss += loss.item()

            # 将损失值放到Iter中，方便观察
            print('Epoch:{}, Step:{}, Loss:{}'.format(epoch, step, loss))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # 损失进行回传
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # 如果步数整除logging_steps，则记录学习率和训练集损失值
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_write.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_write.add_scalar("train_loss", (tr_loss - logging_loss) /
                                    (args.logging_steps * args.gradient_accumulation_steps), global_step)
                logging_loss = tr_loss

            # 如果步数整除eval_steps，则进行模型测试，记录测试集的损失
            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                eval_res = evaluate()
                tb_write.add_scalar("eval_res", eval_res, global_step)
                model.train()

        # 每个epoch进行完，则保存模型
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        # 清空cuda缓存
        torch.cuda.empty_cache()







