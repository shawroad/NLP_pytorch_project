# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 14:03
# @Author  : xiaolu
# @FileName: train.py
# @Software: PyCharm

import numpy as np
import json
import torch
from torch import optim
from torch import nn
import re
import string
from collections import Counter
import os
from model import Model
from sp_model import SPModel
from config import Config
from dataloader import DataIterator

IGNORE_INDEX = -100


def train():
    # 1. 数据集加载
    with open(Config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(Config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(Config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(Config.idx2word_file, 'r') as fh:
        idx2word_dict = json.load(fh)

    train_buckets = [torch.load(Config.train_record_file)]
    dev_buckets = [torch.load(Config.dev_record_file)]

    # (self, buckets, bsz, para_limit, ques_limit, char_limit, shuffle, sent_limit
    def build_train_iterator():
        return DataIterator(train_buckets, Config.batch_size, Config.para_limit, Config.ques_limit, Config.char_limit, True, Config.sent_limit)

    def build_dev_iterator():
        return DataIterator(dev_buckets, Config.batch_size, Config.para_limit, Config.ques_limit, Config.char_limit, False, Config.sent_limit)

    if Config.sp_lambda > 0:
        model = SPModel(word_mat, char_mat)
    else:
        model = Model(word_mat, char_mat)

    print('需要更新的参数量:{}'.format(sum([p.nelement() for p in model.parameters() if p.requires_grad])))
    # 需要更新的参数量:235636

    ori_model = model.to(Config.device)

    lr = Config.init_lr
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.init_lr)

    cur_patience = 0
    global_step = 0
    best_dev_F1 = None
    stop_train = False
    model.train()
    for epoch in range(10000):
        for data in build_train_iterator():
            global_step += 1
            context_idxs = data['context_idxs'].to(Config.device)
            ques_idxs = data['ques_idxs'].to(Config.device)
            context_char_idxs = data['context_char_idxs'].to(Config.device)
            ques_char_idxs = data['ques_char_idxs'].to(Config.device)
            context_lens = data['context_lens'].to(Config.device)

            y1 = data['y1'].to(Config.device)
            y2 = data['y2'].to(Config.device)
            q_type = data['q_type'].to(Config.device)
            is_support = data['is_support'].to(Config.device)
            start_mapping = data['start_mapping'].to(Config.device)
            end_mapping = data['end_mapping'].to(Config.device)
            all_mapping = data['all_mapping'].to(Config.device)
            # print(context_idxs.size())    # torch.Size([2, 942])
            # print(context_char_idxs.size())   # torch.Size([2, 942, 16])
            # print(ques_idxs.size())     # torch.Size([2, 18])
            # print(ques_char_idxs.size())    # torch.Size([2, 18, 16])
            # print(y1.size())    # torch.Size([2])
            # print(y2.size())    # torch.Size([2])
            # print(q_type.size())    # torch.Size([2])
            # print(is_support.size())     # torch.Size([2, 49])
            # print(start_mapping.size())   # torch.Size([2, 942, 49])
            # print(end_mapping.size())    # torch.Size([2, 942, 49])
            # print(all_mapping.size())   # torch.Size([2, 942, 49])

            logit1, logit2, predict_type, predict_support = model(context_idxs, ques_idxs,
                                                                  context_char_idxs, ques_char_idxs,
                                                                  context_lens, start_mapping, end_mapping,
                                                                  all_mapping, return_yp=False)

            loss_1 = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0)
            loss_2 = nll_average(predict_support.view(-1, 2), is_support.view(-1))
            loss = loss_1 + Config.sp_lambda * loss_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('| epoch {:3d} | step {:6d} | lr {:05.5f} | train loss {:8.3f}'.format(epoch, global_step, lr, loss))

            if global_step % 10 == 0:
                model.eval()
                metrics = evaluate_batch(build_dev_iterator(), model, 0, dev_eval_file)
                model.train()

                print('dev loss {:8.3f} | EM {:.4f} | F1 {:.4f}'.format(
                    epoch, metrics['loss'], metrics['exact_match'], metrics['f1']))

                dev_F1 = metrics['f1']
                if best_dev_F1 is None or dev_F1 > best_dev_F1:
                    best_dev_F1 = dev_F1
                    torch.save(ori_model.state_dict(), os.path.join(Config.save, 'model.pt'))
                    cur_patience = 0
                else:
                    cur_patience += 1
                    if cur_patience >= Config.patience:
                        lr /= 2.0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        if lr < Config.init_lr * 1e-2:
                            stop_train = True
                            break
                        cur_patience = 0
        if stop_train:
            break
    print('best_dev_F1 {}'.format(best_dev_F1))


def evaluate_batch(data_source, model, max_batches, eval_file):
    answer_dict = {}
    sp_dict = {}
    total_loss, step_cnt = 0, 0
    iter = data_source
    for step, data in enumerate(iter):
        if step >= max_batches > 0:
            break
        context_idxs = data['context_idxs'].to(Config.device)
        ques_idxs = data['ques_idxs'].to(Config.device)
        context_char_idxs = data['context_char_idxs'].to(Config.device)
        ques_char_idxs = data['ques_char_idxs'].to(Config.device)
        context_lens = data['context_lens'].to(Config.device)
        y1 = data['y1'].to(Config.device)
        y2 = data['y2'].to(Config.device)
        q_type = data['q_type'].to(Config.device)
        is_support = data['is_support'].to(Config.device)
        start_mapping = data['start_mapping'].to(Config.device)
        end_mapping = data['end_mapping'].to(Config.device)
        all_mapping = data['all_mapping'].to(Config.device)

        logit1, logit2, predict_type, predict_support, yp1, yp2 = model(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, context_lens, start_mapping, end_mapping, all_mapping, return_yp=True)
        loss = (nll_sum(predict_type, q_type) + nll_sum(logit1, y1) + nll_sum(logit2, y2)) / context_idxs.size(0) + Config.sp_lambda * nll_average(predict_support.view(-1, 2), is_support.view(-1))
        answer_dict_ = convert_tokens(eval_file, data['ids'], yp1.data.cpu().numpy().tolist(), yp2.data.cpu().numpy().tolist(), np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)
        total_loss += loss.item()
        step_cnt += 1
    loss = total_loss / step_cnt
    metrics = evaluate(eval_file, answer_dict)
    metrics['loss'] = loss
    return metrics


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answer"]
        prediction = value
        assert len(ground_truths) == 1
        cur_EM = exact_match_score(prediction, ground_truths[0])
        cur_f1, _, _ = f1_score(prediction, ground_truths[0])
        exact_match += cur_EM
        f1 += cur_f1
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def convert_tokens(eval_file, qa_id, pp1, pp2, p_type):
    answer_dict = {}
    for qid, p1, p2, type in zip(qa_id, pp1, pp2, p_type):
        if type == 0:
            context = eval_file[str(qid)]["context"]
            spans = eval_file[str(qid)]["spans"]
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer_dict[str(qid)] = context[start_idx: end_idx]
        elif type == 1:
            answer_dict[str(qid)] = 'yes'
        elif type == 2:
            answer_dict[str(qid)] = 'no'
        elif type == 3:
            answer_dict[str(qid)] = 'noanswer'
        else:
            assert False
    return answer_dict


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO_METRIC = (0, 0, 0)
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


if __name__ == '__main__':
    nll_sum = nn.CrossEntropyLoss(reduction='sum', ignore_index=IGNORE_INDEX)
    nll_average = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    nll_all = nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
    train()
