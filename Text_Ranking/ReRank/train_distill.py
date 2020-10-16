# -*- coding: utf-8 -*-
# @Time    : 2020/9/29 9:33
# @Author  : xiaolu
# @FileName: train_distill.py
# @Software: PyCharm
import os
import gzip
import torch
import random
import pickle
import time
from torch import nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score
from transformers import BertTokenizer
from sklearn.metrics import precision_score
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
# from model import Model
from distill_model import CModel
from model import Model
from config import set_args


class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_id, scores=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.scores = scores

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "input_ids: %s" % (self.input_ids)
        s += ", input_mask: %s" % (self.input_mask)
        s += ", segment_ids: %s" % (self.segment_ids)
        s += ", label_id: %s" % (self.label_id)
        s += ", scores: %s" % (self.scores)
        return s


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def evaluate(epoch):
    print("***** Running evaluating *****")
    print("  Num examples = {}".format(len(eval_features)))
    print("  Batch size = {}".format(args.eval_batch_size))

    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    # eval_scores = torch.tensor([f.scores for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    student_model.eval()
    eval_loss = 0
    step = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc='Evaluation'):
        step += 1
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            t_logits, layer_13_output = teacher_model(input_ids, input_mask, segment_ids)
            logits, layer_3_output = student_model(input_ids, input_mask, segment_ids)

        loss = compute_loss(logits, label_ids, t_logits, layer_13_output, layer_3_output)
        eval_loss += loss.mean().item()  # 统计一个batch的损失 一个累加下去

        labels = label_ids.data.cpu().numpy()
        predic = torch.max(logits.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

    # 损失 召回率 查准率
    eval_loss = eval_loss / step
    eval_accuracy = accuracy_score(labels_all, predict_all)
    eval_recall = recall_score(labels_all, predict_all)
    eval_precision = precision_score(labels_all, predict_all)
    s = 'epoch:{}, eval_loss: {}, eval_precision: {}, eval_accuracy:{}, eval_recall:{}'.format(epoch, eval_loss, eval_precision, eval_accuracy, eval_recall)
    print(s)
    s += '\n'
    with open('result_distill_big.txt', 'a+') as f:
        f.write(s)
    return eval_loss, eval_accuracy


def att_mse_loss(attention_S, attention_T, mask=None):
    if mask is None:
        attention_S_select = torch.where(attention_S <= -1e-3, torch.zeros_like(attention_S), attention_S)
        attention_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), attention_T)
        loss = F.mse_loss(attention_S_select, attention_T_select)
    else:
        mask = mask.to(attention_S).unsqueeze(1).expand(-1, attention_S.size(1), -1) # (bs, num_of_heads, len)
        valid_count = torch.pow(mask.sum(dim=2),2).sum()
        loss = (F.mse_loss(attention_S, attention_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(2)).sum() / valid_count
    return loss


def kd_mse_loss(logits_S, logits_T, temperature=1):
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    loss = F.mse_loss(beta_logits_S, beta_logits_T)
    return loss


def kd_ce_loss(logits_S, logits_T, temperature=1):
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss


def compute_loss(logits, labels_ids, scores, layer_13_output, layer_3_output):
    scores = scores.softmax(dim=1)
    # 计算两种损失
    # loss1 = F.cross_entropy(logits.view(-1, 2), labels_ids.view(-1))   # 分类损失
    loss2 = kd_mse_loss(logits, scores)   # 和teacher_model 的logits计算的损失

    # print(len(layer_13_output))   # 13  前面的第一层加入了embedding
    # print(len(layer_3_output))

    # 1, 2, 3 -> 1, 6, 12
    # 1-1
    layer1_mse_loss = att_mse_loss(layer_3_output[0], layer_13_output[1])
    # 2-6
    layer2_mse_loss = att_mse_loss(layer_3_output[1], layer_13_output[7])
    # 3-12
    layer3_mse_loss = att_mse_loss(layer_3_output[2], layer_13_output[-1])
    loss = loss2 + layer1_mse_loss + layer2_mse_loss + layer3_mse_loss
    
   
    # loss = args.alpha * loss1 + (1 - args.alpha) * loss2
    return loss


if __name__ == '__main__':
    args = set_args()
    set_seed(args)  # 设定随机种子

    device = torch.device('cuda: {}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # 加载训练集
    # with gzip.open(args.train_features_path, 'rb') as f:
    #     train_features = pickle.load(f)
    # 加载训练集
    with gzip.open(args.train_features_path, 'rb') as f:
        train_features = pickle.load(f)

    with gzip.open(args.eval_features_path, 'rb') as f:
        eval_features = pickle.load(f)


    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # 模型
    student_model = CModel(device)

    # 加载老师模型  选一个最优的
    teacher_model = Model()
    teacher_model.load_state_dict(torch.load('./save_teacher_model/epoch3_ckpt.bin'))
    teacher_model = teacher_model.to(device)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_file)
    param_optimizer = list(student_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = 0.05 * num_train_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    ce_loss = nn.NLLLoss()   # 分类损失
    mse_loss = nn.MSELoss()    # 均方误差损失

    best_loss = None
    global_step = 0
    student_model = student_model.to(device)

    if args.do_train:
        print("***** Running training *****")
        print("  Num examples = {}".format(len(train_features)))
        print("  Batch size = {}".format(args.train_batch_size))
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        # all_scores = torch.tensor([f.scores for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        student_model.train()
        for epoch in range(args.num_train_epochs):
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
            for step, batch in enumerate(train_dataloader):
                start_time = time.time()
                batch = tuple(t.to(device) for t in batch)
                # input_ids, input_mask, segment_ids, labels_ids = batch
                input_ids, input_mask, segment_ids, labels_ids = batch
                # scores = scores.softmax(dim=1)
                # 先用老师模型把各层的输出得到
                with torch.no_grad():
                    t_logits, layer_13_output = teacher_model(input_ids, input_mask, segment_ids)

                logits, layer_3_output = student_model(input_ids, input_mask, segment_ids)

                loss = compute_loss(logits, labels_ids, t_logits, layer_13_output, layer_3_output)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                print('<distilling> epoch:{}, step:{}, loss:{:10f}, time_cost:{:10f}'.format(epoch, step, loss, time.time()-start_time))
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                # test_loss, test_acc = evaluate(epoch)

            # 验证验证集
            test_loss, test_acc = evaluate(epoch)
            # 验证训练集中四万之后的数据
            student_model.train()

            if best_loss is None or best_loss > test_loss:
                best_loss = test_loss
                model_to_save = student_model.module if hasattr(student_model, 'module') else student_model  # Only save the model it-self
                os.makedirs(args.ckpt_dir, exist_ok=True)
                output_model_file = os.path.join(args.save_student_model, "best_pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

            # Save a trained model
            model_to_save = student_model.module if hasattr(student_model, 'module') else student_model  # Only save the model it-self
            output_model_file = os.path.join(args.save_student_model, "epoch{}_ckpt.bin".format(epoch))
            torch.save(model_to_save.state_dict(), output_model_file)

