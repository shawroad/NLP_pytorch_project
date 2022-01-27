"""
@file   : run_cls.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-01-05
"""
import os
import copy
import torch
import random
from tqdm import tqdm
import numpy as np
from config import set_args
from model_cls import Model
from sklearn import metrics
from focal_loss import BCEFocalLoss
from torch.utils.data import DataLoader
from transformers.models.bert import BertTokenizer
from data_helper import load_data, CustomDataset, collate_fn
from transformers import AdamW, get_linear_schedule_with_warmup


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


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
    eval_targets = []
    eval_predict = []
    model.eval()
    for step, batch in tqdm(enumerate(val_dataloader)):
        input_ids, input_mask, segment_ids, label_ids = batch
        if torch.cuda.is_available():
            input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
            label_ids = label_ids.cuda()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=input_mask, segment_ids=segment_ids)
        eval_targets.extend(label_ids.cpu().detach().numpy().tolist())
        eval_predict.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
    eval_accuracy, eval_f1_score_micro, eval_f1_score_macro = get_metrics(eval_predict, eval_targets)
    return eval_accuracy, eval_f1_score_micro, eval_f1_score_macro


if __name__ == '__main__':
    args = set_args()
    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    # 加载数据集
    train_df, val_df = load_data(args.train_data)
    print('训练集的大小:', train_df.shape)
    print('验证集的大小:', val_df.shape)

    # 训练数据集准备
    train_dataset = CustomDataset(dataframe=train_df, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn, num_workers=2)

    # 验证集准备
    val_dataset = CustomDataset(dataframe=val_df, tokenizer=tokenizer)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=args.val_batch_size,
                                collate_fn=collate_fn, num_workers=2)

    total_steps = len(train_dataloader) * args.num_train_epochs

    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    model = Model()

    if torch.cuda.is_available():
        model.cuda()

    optimizer_grouped_parameters = [
        {"params": model.nezha.parameters()},
        {'params': model.highway.parameters(), 'lr': args.learning_rate * 10},
        {"params": model.classifier.parameters(), 'lr': args.learning_rate * 10}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)

    # loss_func = nn.BCEWithLogitsLoss()   # 普通的BCE损失
    loss_func = BCEFocalLoss()

    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Batch size = %d" % args.train_batch_size)
    print("  Num steps = %d" % num_train_optimization_steps)
    for epoch in range(args.num_train_epochs):
        model.train()
        train_label, train_predict = [], []
        epoch_loss = 0

        # 制作tqdm对象
        step = 0
        pbar = tqdm(train_dataloader, colour='yellow')  # color可以指定进度条的颜色
        for batch in pbar:
            step += 1
            # for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch
            if torch.cuda.is_available():
                input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
                label_ids = label_ids.cuda()

            logits = model(input_ids=input_ids, attention_mask=input_mask, segment_ids=segment_ids)
            loss = loss_func(logits, label_ids)
            loss.backward()
            pbar.set_description("当前轮次:{}, 正在迭代:{}/{}, Loss:{:10f}".format(epoch, step, len(train_dataloader), loss))  # 在进度条前面定义一段文字
            # print('epoch:{}, step:{}, loss:{:10f}'.format(epoch, step, loss))
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            epoch_loss += loss

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_label.extend(label_ids.cpu().detach().numpy().tolist())
            train_predict.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())

        train_accuracy, train_f1_score_micro, train_f1_score_macro = get_metrics(train_predict, train_label)
        eval_accuracy, eval_f1_score_micro, eval_f1_score_macro = evaluate()

        s = 'Epoch: {} | Loss: {:10f} | Train acc: {:10f} | Train f1_score_micro: {:10f} | Train f1_score_macro: {:10f} | Val acc: {:10f} | Val f1_score_micro: {:10f} | Val f1_score_macro: {:10f}'
        ss = s.format(epoch, epoch_loss / len(train_dataloader), train_accuracy, train_f1_score_micro, train_f1_score_macro, eval_accuracy, eval_f1_score_micro, eval_f1_score_macro)

        logs_path = os.path.join(args.output_dir, 'logs.txt')
        with open(logs_path, 'a+') as f:
            ss += '\n'
            f.write(ss)

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "base_model_epoch_{}.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)
