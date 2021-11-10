"""
@file   : run_RDrop.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-11-08
"""
'''
注: 看此代码直接从151行看起。  其他都是分类相关的预处理模型等。
RDrop不同之处在于加入了logits之间的KL_Loss
'''
import os
import copy
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from sklearn import metrics
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import set_args
from transformers.models.bert import BertTokenizer
from data_helper import load_data, CustomDataset, collate_fn
from model_base import Model
from transformers import AdamW, get_linear_schedule_with_warmup


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
    model.eval()
    for step, batch in tqdm(enumerate(eval_dataloader)):
        input_ids, input_mask, segment_ids, label_ids = batch
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            label_ids = label_ids.cuda()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=input_mask)
            # teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)
            # outputs = teacher_model(input_ids, segment_ids, input_mask)
        eval_targets.extend(label_ids.cpu().detach().numpy().tolist())
        eval_predict.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
    eval_accuracy, eval_f1_score_micro, eval_f1_score_macro = get_metrics(eval_predict, eval_targets)
    return eval_accuracy, eval_f1_score_micro, eval_f1_score_macro


def calc_loss(logits1, logits2, label):
    # 分类损失
    loss_fct = nn.BCEWithLogitsLoss()
    bce_loss = 0.5 * (loss_fct(logits1, label) + loss_fct(logits2, label))

    # KL散度损失
    loss1 = F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='none')
    loss2 = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1), reduction='none')
    loss1 = loss1.sum()
    loss2 = loss2.sum()
    kl_loss = (loss1 + loss2) / 2
    return bce_loss, kl_loss


if __name__ == "__main__":
    args = set_args()
    args.train_batch_size = 8
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    num_labels = args.num_labels

    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case)

    # 加载数据
    train_df, val_df = load_data(args.train_data)
    # print(train_df.shape)   # (1013458, 2)
    # print(val_df.shape)   # (53340, 2)
    train_dataset = CustomDataset(dataframe=train_df, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    total_steps = len(train_dataloader) * args.num_train_epochs

    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    # teacher_model.load_state_dict(torch.load('./outputs/model_epoch_3.bin'))
    model = Model()
    # model.load_state_dict(torch.load('./outputs/xxx.bin'))

    if torch.cuda.is_available():
        model.cuda()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)

    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Batch size = %d" % args.train_batch_size)
    print("  Num steps = %d" % num_train_optimization_steps)
    for epoch in range(args.num_train_epochs):
        model.train()
        train_targets = []
        train_predict = []
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                input_mask = input_mask.cuda()
                segment_ids = segment_ids.cuda()
                label_ids = label_ids.cuda()

            # 注意，RDrop主要不同在一下几点。
            # 1. 将同一匹数据 分别通过两次模型  输出不一定一样 因为dropout等信息不同
            logits1, _, _ = model(input_ids=input_ids, attention_mask=input_mask)
            logits2, _, _ = model(input_ids=input_ids, attention_mask=input_mask)

            # 2. 在正常的分类上，加上两个logits之间的KL_Loss
            bce_loss, kl_loss = calc_loss(logits1, logits2, label_ids)

            loss = 4 * kl_loss + bce_loss

            print('epoch:{}, step:{}, total_loss:{:10f}, kl_loss:{:10f}, bce_loss:{:10f}'.format(epoch, step, loss,
                                                                                                 kl_loss, bce_loss))

            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            epoch_loss += loss

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_targets.extend(label_ids.cpu().detach().numpy().tolist())
            train_predict.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())

        train_accuracy, train_f1_score_micro, train_f1_score_macro = get_metrics(train_predict, train_targets)
        s = 'Epoch: {} | Loss: {:10f} | Train acc: {:10f} | Train f1_score_micro: {:10f} | Train f1_score_macro: {:10f} | Val acc: {:10f} | Val f1_score_micro: {:10f} | Val f1_score_macro: {:10f}'
        eval_accuracy, eval_f1_score_micro, eval_f1_score_macro = evaluate()
        ss = s.format(epoch, epoch_loss / len(train_dataloader), train_accuracy, train_f1_score_micro,
                      train_f1_score_macro, eval_accuracy, eval_f1_score_micro, eval_f1_score_macro)
        logs_path = os.path.join(args.output_dir, 'logs.txt')
        with open(logs_path, 'a+') as f:
            ss += '\n'
            f.write(ss)

        # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "base_model_epoch_{}.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)
