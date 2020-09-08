# -*- coding: utf-8 -*-
# @Time    : 2020/8/26 17:21
# @Author  : xiaolu
# @FileName: train.py
# @Software: PyCharm
import os
import torch
import json
import time
from torch import nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from rlog import _log_normal, _log_warning, _log_info, _log_error, _log_toomuch, _log_bg_blue, _log_bg_pp, _log_fg_yl, \
    _log_fg_cy, _log_black, rainbow
from model import Model
from config import Config



def evaluate(dev_features, model):
    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(dev_features)))
    print("  Batch size = {}".format(Config.eval_batch_size))

    eval_input_ids = torch.tensor([f['input_ids'] for f in dev_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f['attention_mask'] for f in dev_features], dtype=torch.long)
    eval_segment_ids = torch.tensor([f['token_type_ids'] for f in dev_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f['labels'] for f in dev_features], dtype=torch.long)

    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=Config.eval_batch_size)

    model.eval()
    eval_loss = 0
    step = 0

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc='Evaluation'):
        step += 1
        input_ids = input_ids.to(Config.device)
        input_mask = input_mask.to(Config.device)
        segment_ids = segment_ids.to(Config.device)
        label_ids = label_ids.to(Config.device)
        with torch.no_grad():
            loss, logits = model(input_ids, input_mask, segment_ids, labels=label_ids)

        eval_loss += loss.mean().item()   # 统计一个batch的损失 一个累加下去

        labels = label_ids.data.cpu().numpy()
        predic = torch.max(logits.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

    # 损失 召回率 查准率
    eval_loss = eval_loss / step
    eval_accuracy = accuracy_score(labels_all, predict_all)  # 准确率

    s = 'epoch:{}, eval_loss: {}, eval_accuracy:{}'.format(epc, eval_loss, eval_accuracy)
    print(s)
    s += '\n'
    with open('result_eval.txt', 'a+') as f:
        f.write(s)
    return eval_loss, eval_accuracy


if __name__ == '__main__':
    # 1.加载训练集
    train_features = []
    with open('./work_dir/train_processed.work_dir', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = json.loads(line)
            train_features.append(line)

    # 2. 加载验证集
    dev_features = []
    with open('./work_dir/dev_processed.work_dir', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = json.loads(line)
            dev_features.append(line)

    all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['attention_mask'] for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['token_type_ids'] for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f['labels'] for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    model = Model()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = int(
        len(train_features) / Config.train_batch_size / Config.gradient_accumulation_steps * Config.num_train_epochs)
    warmup_steps = 0.05 * t_total
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    best = 0.0
    for epc in range(Config.num_train_epochs):
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=Config.train_batch_size)

        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            batch = tuple(t.to(Config.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            loss, logits = model(input_ids, input_mask, segment_ids, label_ids)

            labels = label_ids.data.cpu().numpy()
            predic = torch.max(logits.data, 1)[1].cpu().numpy()

            train_accuracy = accuracy_score(labels, predic)  # 准确率

            if Config.gradient_accumulation_steps > 1:
                loss = loss / Config.gradient_accumulation_steps

            s = 'epoch:{}, step:{}, loss:{:10f}, accuracy:{:10f}, time_cost:{:10f}'.format(epc, step, loss,
                                                                                           train_accuracy,
                                                                                           time.time() - start_time)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

            if (step + 1) % Config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            eval_loss, eval_accuracy = evaluate(dev_features, model)
            print(eval_accuracy)
            exit()

        if epc % 1 == 0:
            eval_loss, eval_accuracy = evaluate(dev_features, model)
            if eval_accuracy > best:
                best = eval_accuracy
                # 保存模型
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(Config.ckpt_dir, "best_pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(Config.ckpt_dir, "epoch{}_ckpt.bin".format(epc))
        torch.save(model_to_save.state_dict(), output_model_file)






