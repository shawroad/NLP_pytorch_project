"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-22
"""
import os
import torch
import time
import copy
import numpy as np
from tqdm import tqdm
from model import Model
from pdb import set_trace
from sklearn import metrics
from config import set_args
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from data_helper import load_data, CustomDataset
from roformer.tokenization_roformer import RoFormerTokenizer
from transformers.models.bert import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    # 将序列padding到最大长度
    if max_len > 512:
        max_len = 512
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len-len(input_ids))
    return input_ids


def collate_fn(batch):
    # 获取当前batch中最大长度
    max_len = max([len(d['input_ids']) for d in batch])
    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []
    for dat in batch:
        input_ids.append(pad_to_maxlen(dat['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(dat['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(dat['token_type_ids'], max_len=max_len))
        labels.append(dat['targets'])
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.float)
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids


def get_metrics(outputs, targets):
    """
    return metrics for top1 prediction
    """
    max_values = np.array(outputs).max(axis=1)
    outputs_top1 = copy.deepcopy(outputs)
    for i in range(len(outputs_top1)):
        outputs_top1[i] = outputs_top1[i] == max_values[i]
    targets = np.array(targets)
    outputs_top1 = np.array(outputs_top1)
    accuracy = metrics.accuracy_score(targets, outputs_top1)
    f1_score_micro = metrics.f1_score(targets, outputs_top1, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs_top1, average='macro')
    return accuracy, f1_score_micro, f1_score_macro


def evaluate():
    eval_dataset = CustomDataset(dataframe=val_df, tokenizer=tokenizer)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    eval_targets = []
    eval_predict = []
    for step, batch in tqdm(enumerate(eval_dataloader)):
        input_ids, input_mask, segment_ids, label_ids = batch
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            label_ids = label_ids.cuda()

        with torch.no_grad():
            outputs = model(input_ids, input_mask, segment_ids)
        eval_targets.extend(label_ids.cpu().detach().numpy().tolist())
        eval_predict.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    eval_accuracy, eval_f1_score_micro, eval_f1_score_macro = get_metrics(train_predict, train_targets)
    return eval_accuracy, eval_f1_score_micro, eval_f1_score_macro


if __name__ == '__main__':
    args = set_args()

    os.makedirs(args.outputs, exist_ok=True)  # 所有输出都在这个文件夹中

    # 1. 实例化模型
    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    # 2. 加载数据集
    tokenizer = RoFormerTokenizer.from_pretrained('./roformer_pretrain/vocab.txt')

    train_df, val_df = load_data(args.train_data)
    # print(train_df.shape)   # (1013458, 2)
    # print(val_df.shape)   # (53340, 2)
    mydataset = CustomDataset(dataframe=train_df, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=mydataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)

    # 3. 损失函数
    loss_fct = BCEWithLogitsLoss()

    # 4. 优化器
    optimizer = AdamW([
        {'params': model.roformer.parameters()},
        {'params': model.classifier.parameters(), 'lr': args.learning_rate*10}],
        lr=args.learning_rate)

    total_steps = len(train_dataloader) * args.num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps, num_training_steps=total_steps)

    for epoch in range(args.num_epochs):
        start_time = time.time()
        epoch_loss = 0
        model.train()
        train_targets = []
        train_predict = []
        for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                segment_ids = segment_ids.cuda()
                label_ids = label_ids.cuda()

            outputs = model(input_ids, input_mask, segment_ids)
            loss = loss_fct(outputs, label_ids)
            print('epoch:{}, step:{}, loss:{}'.format(epoch, step, loss))

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_targets.extend(label_ids.cpu().detach().numpy().tolist())
            train_predict.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())


        train_accuracy, train_f1_score_micro, train_f1_score_macro = get_metrics(train_predict, train_targets)
        eval_accuracy, eval_f1_score_micro, eval_f1_score_macro = evaluate()
        s = 'Epoch: {} | Loss: {:10f} | Train acc: {:10f} | Train f1_score_micro: {:10f} | Train f1_score_macro: {:10f} | Val acc: {:10f} | Val f1_score_micro: {:10f} | Val f1_score_macro: {:10f}'
        print(s.format(epoch, epoch_loss, train_accuracy, train_f1_score_micro, train_f1_score_macro, eval_accuracy, eval_f1_score_micro, eval_f1_score_macro))

        logs_path = os.path.join(args.outputs, 'logs.txt')
        with open(logs_path, 'a+') as f:
            s += '\n'
            f.write(s)

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.outputs, "model_epoch_{}.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)

