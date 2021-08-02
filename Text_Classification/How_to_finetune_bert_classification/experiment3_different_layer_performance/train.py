"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-30$
"""
import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from config import set_args
from pdb import set_trace
from data_helper import load_data
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from model import BertForSequenceClassification


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y in tqdm(test_loader):
            if torch.cuda.is_available():
                cur_input_ids = cur_input_ids.cuda()
                cur_attention_mask = cur_attention_mask.cuda()
                cur_token_type_ids = cur_token_type_ids.cuda()
                cur_y = cur_y.cuda()
            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)[0]
            _, predicted = torch.max(outputs, 1)
            total += cur_y.size(0)
            correct += (predicted == cur_y).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    args = set_args()
    set_seed(args.seed)

    # 1. 实例化模型
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = BertForSequenceClassification.from_pretrained(args.bert_path, num_labels=2)

    if torch.cuda.is_available():
        model.cuda()

    # 2. 加载数据集
    # 训练集
    train_path = os.path.join("../data", 'train.csv')
    train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(train_path, tokenizer)
    train_data = TensorDataset(train_input_ids, train_attention_mask, train_token_type_ids, y_train)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # 测试集
    test_path = os.path.join("../data", 'test.csv')
    test_input_ids, test_attention_mask, test_token_type_ids, y_test = load_data(test_path, tokenizer)
    test_data = TensorDataset(test_input_ids, test_attention_mask, test_token_type_ids, y_test)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # 3. 定义优化器
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]   # 后面weight_decay: 权重衰减的系数

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    total_step = len(train_loader) * args.num_epochs // args.gradient_accumulation_step

    scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps=total_step * args.warm_up_proportion,
                                                num_training_steps=total_step)
    loss_fct = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(args.num_epochs):
        model.train()
        model.zero_grad()
        for step, (cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y) in enumerate(train_loader):
            if torch.cuda.is_available():
                cur_input_ids = cur_input_ids.cuda()
                cur_attention_mask = cur_attention_mask.cuda()
                cur_token_type_ids = cur_token_type_ids.cuda()
                cur_y = cur_y.cuda()

            outputs = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)[0]
            loss = loss_fct(outputs, cur_y)
            print('Epoch:{}, Step:{}, Loss:{:10f}'.format(epoch, step, loss))

            if args.gradient_accumulation_step > 1:
                loss /= args.gradient_accumulation_step

            loss.backward()

            if (step + 1) % args.gradient_accumulation_step == 0:
                # nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        val_acc = evaluate()
        s = 'epoch: {}, test_acc:{}'.format(epoch, val_acc)
        log_file = os.path.join(args.output_dir, 'log.txt')
        with open(log_file, 'a+', encoding='utf8') as f:
            s += '\n'
            f.write(s)

        mode_save_file = os.path.join(args.output_dir, "epoch_{}.bin".format(epoch))
        torch.save(model.state_dict(), mode_save_file)

        if val_acc > best_acc:
            best_model_save_file = os.path.join(args.output_dir, 'best_model.bin')
            torch.save(model.state_dict(), best_model_save_file)
            best_acc = val_acc
