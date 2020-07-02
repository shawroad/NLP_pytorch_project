"""

@file  : train.py

@author: xiaolu

@time  : 2020-04-09

"""
import torch
import random
import time
import datetime
import numpy as np
from model import Model
from DataLoader import DatasetIterater, build_dataset
from config import Config
from rlog import _log_normal, _log_warning, _log_info, _log_error, _log_toomuch, _log_bg_blue, _log_bg_pp, _log_fg_yl, \
    _log_fg_cy, _log_black, rainbow
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import json

# 随机种子
random.seed(Config.seed)
torch.manual_seed(Config.seed)

train_loss = []
eval_loss = []


def train():
    device = Config.device
    # 准备数据
    train_data, dev_data = build_dataset(Config)
    train_iter = DatasetIterater(train_data, Config)
    dev_iter = DatasetIterater(dev_data, Config)

    model = Model().to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 这里我们用bertAdam优化器  
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.learning_rate, correct_bias=False)  # 要重现BertAdam特定的行为，请设置correct_bias = False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05, num_training_steps=len(train_iter) * Config.num_epochs)  # PyTorch调度程序用法如下：



    model.to(device)
    model.train()

    best_loss = 100000.0
    for epoch in range(Config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, Config.num_epochs))
        for step, batch in enumerate(train_iter):
            start_time = time.time()
            ids, input_ids, input_mask, start_positions, end_positions = \
                batch[0], batch[1], batch[2], batch[3], batch[4]
            input_ids, input_mask, start_positions, end_positions = \
                input_ids.to(device), input_mask.to(device), start_positions.to(device), end_positions.to(device)

            # print(input_ids.size())
            # print(input_mask.size())
            # print(start_positions.size())
            # print(end_positions.size())

            loss, _, _ = model(input_ids, attention_mask=input_mask,
                               start_positions=start_positions, end_positions=end_positions)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=20)
            optimizer.step()
            scheduler.step()

            time_str = datetime.datetime.now().isoformat()
            log_str = 'time:{}, epoch:{}, step:{}, loss:{:8f}, spend_time:{:6f}'.format(time_str, epoch, step, loss,
                                                                                        time.time() - start_time)
            rainbow(log_str)

            train_loss.append(loss)

        if epoch % 1 == 0:
            eval_loss = valid(model, dev_iter)
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(), './save_model/' + 'best_model.bin')
                model.train()


def valid(model, eval_file):
    '''
    验证数据集
    :param model:
    :param dataset:
    :param eval_file:
    :return:
    '''
    device = Config.device
    model.eval()
    answer_dict = {}
    losses = []

    with open('./data/dev.data', 'r') as f:
        data = dict()
        lines = f.readlines()
        for line in lines:
            source = json.loads(line.strip())
            ids = source['ids']
            data[ids] = source

    with torch.no_grad():
        for step, batch in enumerate(eval_file):
            ids, input_ids, input_mask, start_positions, end_positions = \
                batch[0], batch[1], batch[2], batch[3], batch[4]
            input_ids, input_mask, start_positions, end_positions = \
                input_ids.to(device), input_mask.to(device), start_positions.to(device), end_positions.to(device)
            loss, pred_start, pred_end = model(input_ids, attention_mask=input_mask,
                                               start_positions=start_positions, end_positions=end_positions)
            losses.append(loss.item())

            # pred_start's size --> (batch_size, max_len)
            yp1 = torch.argmax(pred_start, 1)
            yp2 = torch.argmax(pred_end, 1)
            yps = torch.stack([yp1, yp2], dim=1)   # (batch_size, 2, 1)
            ymin, _ = torch.min(yps, 1)    # 挑小的做起始值
            ymax, _ = torch.max(yps, 1)    # 挑大的做结束值
            answer_dict_, _ = convert_tokens(data, ids.tolist(), ymin.tolist(), ymax.tolist())
            answer_dict.update(answer_dict_)

    loss = np.mean(losses)
    metrics = evaluate(data, answer_dict)
    metrics["loss"] = loss
    res = "VALID loss {:8f} F1 {:8f} EM {:8f}\n".format(loss, metrics["f1"], metrics["exact_match"])
    print(res)
    res += '\n'
    with open('log.txt', 'a+') as f:
        f.write(res)
    return loss


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        item = eval_file[key]
        context = item["input_ids"]
        ground_truths = context[item['start_position']: item['end_position']]
        ground_truths = tokenizer.decode(ground_truths)
        prediction = value
        # print(prediction)
        # print(ground_truths)
        f1 += calc_f1_score(ground_truths, prediction)
        exact_match += calc_em_score(ground_truths, prediction)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def calc_f1_score(answers, prediction):
    if len(prediction) == 0:
        return 0
    lcs, lcs_len = find_lcs(answers, prediction)
    precision = 1.0 * lcs_len / len(prediction)
    recall = 1.0 * lcs_len / len(answers)
    if precision == 0 and recall == 0:
        return 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


def find_lcs(s1, s2):
    # find longest common string
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


def calc_em_score(answers, prediction):
    em = 0
    ans_ = answers
    prediction_ = prediction
    if ans_ == prediction_:
        em = 1
    return em


def convert_tokens(eval_file, qa_id, pp1, pp2):
    '''
    找出起始和结束　然后输出答案那一截
    :param eval_file:
    :param qa_id:
    :param pp1:
    :param pp2:
    :return:
    '''
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        item = eval_file[qid]
        context = item["input_ids"]
        start_idx = p1
        end_idx = p2
        uuid = item["ids"]
        if p1 > p2 or p1 > len(context) or p2 > len(context):
            ans = ""
        else:
            ans = context[start_idx: end_idx]
            ans = tokenizer.decode(ans)
        answer_dict[qid] = ans
        remapped_dict[uuid] = ans

    return answer_dict, remapped_dict


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('./albert_pretrain/vocab.txt')
    train()

