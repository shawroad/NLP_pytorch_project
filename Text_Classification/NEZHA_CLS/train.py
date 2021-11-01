"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-09-30
"""
import torch
from config import set_args
from torch import nn
from model import Model
from torch.utils.data import DataLoader
from data_utils import TxtDataSet, BlockShuffleDataLoader, collate_fn
from transformers import AdamW, get_linear_schedule_with_warmup


if __name__ == '__main__':
    args = set_args()

    # 1. 加载训练集
    train_dataset = TxtDataSet(data_set_name='train', path='./data/train.json')
    # 排序 按batch进行shuffle
    train_data_loader = BlockShuffleDataLoader(train_dataset,
                                               sort_key=lambda x: len(x["input_ids"]),
                                               is_shuffle=True,
                                               batch_size=args.train_batch_size,
                                               collate_fn=collate_fn)
    # 2. 加载验证集
    dev_dataset = TxtDataSet(data_set_name='dev', path='./data/test.json')
    dev_data_loader = DataLoader(dev_dataset, shuffle=False,
                                 batch_size=args.train_batch_size, collate_fn=collate_fn)

    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)

    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    no_decay = ["bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    loss_fct = nn.CrossEntropyLoss()

    model.train()
    tr_loss, logging_loss, max_acc = 0.0, 0.0, 0.0
    for epoch in range(10):
        for step, batch in enumerate(train_data_loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            label = batch["label"]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                label = label.cuda()

            logits = model(input_ids, attention_mask)
            loss = loss_fct(logits, label)
            print('epoch:{}, step:{}, loss:{:10f}'.format(epoch, step, loss))
            exit()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
