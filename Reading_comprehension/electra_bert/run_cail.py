# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 16:29
# @Author  : xiaolu
# @FileName: train.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 10:21
# @Author  : xiaolu
# @FileName: train.py
# @Software: PyCharm

import argparse
from os.path import join
from data_helper import DataHelper
from config import set_config
from tqdm import tqdm
from transformers import BertModel
from transformers import BertConfig
from model import BertSupportNet
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import convert_to_tokens
from data_iterator_pack import IGNORE_INDEX
import numpy as np
import random
from data_process import InputFeatures, Example
from config import Config
import json
import torch
from torch import nn


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def dispatch(context_encoding, context_mask, batch, device):
    batch['context_encoding'] = context_encoding.cuda(device)
    batch['context_mask'] = context_mask.float().cuda(device)
    return batch


def compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position):

    loss1 = criterion(start_logits, batch['y1']) + criterion(end_logits, batch['y2'])
    loss2 = args.type_lambda * criterion(type_logits, batch['q_type'])

    sent_num_in_batch = batch["start_mapping"].sum()
    # print(batch['start_mapping'].size())   # torch.Size([4, 1, 512])
    # print(sent_num_in_batch.size())   # torch.Size([])
    # print(sent_num_in_batch)  # tensor(4.)

    # is_support  size: batch_size, sen_limit

    loss3 = args.sp_lambda * sp_loss_fct(sp_logits.view(-1), batch['is_support'].float().view(-1)).sum() / sent_num_in_batch
    loss = loss1 + loss2 + loss3
    return loss, loss1, loss2, loss3


@torch.no_grad()
def predict(model, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):
    model.eval()
    answer_dict = {}
    sp_dict = {}
    dataloader.refresh()
    total_test_loss = [0] * 5

    for batch in tqdm(dataloader):

        batch['context_mask'] = batch['context_mask'].float()
        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)

        loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)

        for i, l in enumerate(loss_list):
            if not isinstance(l, int):
                total_test_loss[i] += l.item()

        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                         start_position.data.cpu().numpy().tolist(),
                                         end_position.data.cpu().numpy().tolist(),
                                         np.argmax(type_logits.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = batch['ids'][i]

            cur_sp_logit_pred = []  # for sp logit output
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if need_sp_logit_file:
                    temp_title, temp_id = example_dict[cur_id].sent_names[j]
                    cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                if predict_support_np[i, j] > args.sp_threshold:
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            sp_dict.update({cur_id: cur_sp_pred})

    new_answer_dict = {}
    for key, value in answer_dict.items():
        new_answer_dict[key] = value.replace(" ", "")
    prediction = {'answer': new_answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w', encoding='utf8') as f:
        json.dump(prediction, f, indent=4, ensure_ascii=False)

    for i, l in enumerate(total_test_loss):
        print("Test Loss{}: {}".format(i, l / len(dataloader)))
    test_loss_record.append(sum(total_test_loss[:3]) / len(dataloader))


def train_epoch(data_loader, model, predict_during_train=False):
    model.train()
    # pbar = tqdm(total=len(data_loader))
    epoch_len = len(data_loader)
    step_count = 0
    predict_step = epoch_len // 5
    while not data_loader.empty():   # data_loader.empty()输出True 表示一轮结束
        step_count += 1
        batch = next(iter(data_loader))
        batch['context_mask'] = batch['context_mask'].float()
        train_batch(model, batch)
        del batch

        # if predict_during_train and (step_count % predict_step == 0):
        #     predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
        #             join(args.prediction_path, 'pred_seed_{}_epoch_{}_{}.json'.format(args.seed, epc, step_count)))
        #     model_to_save = model.module if hasattr(model, 'module') else model
        #     torch.save(model_to_save.state_dict(),
        #                join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_{}.pth".format(args.seed, epc, step_count)))
        #     model.train()
        # pbar.update(1)
    predict(model, eval_dataset, dev_example_dict, dev_feature_dict,
            join(args.prediction_path, 'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)))
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(),
               join(args.checkpoint_path, "ckpt_seed_{}_epoch_{}_99999.pth".format(args.seed, epc)))


def train_batch(model, batch):
    # batch是数据
    global global_step, total_train_loss

    start_logits, end_logits, type_logits, sp_logits, start_position, end_position = model(batch)
    loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)
    loss_list = list(loss_list)
    if args.gradient_accumulation_steps > 1:
        loss_list[0] = loss_list[0] / args.gradient_accumulation_steps

    loss_list[0].backward()

    if (global_step + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    global_step += 1

    for i, l in enumerate(loss_list):
        if not isinstance(l, int):
            total_train_loss[i] += l.item()

    print('epoch: {}, step: {}, start_end_loss: {:6f},  four_cls_loss: {:6f}, sp_loss: {:6f}, sum_loss: {:6f}'.format(
        epc,
        global_step,
        loss_list[0].item(),
        loss_list[1].item(),
        loss_list[2].item(),
        loss_list[3].item()
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = set_config()

    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type    # 2

    if args.seed == 0:
        args.seed = random.randint(0, 100)
        set_seed(args)

    # 加载数据集
    Full_Loader = helper.train_loader   # 加载训练集
    dev_example_dict = helper.dev_example_dict
    dev_feature_dict = helper.dev_feature_dict
    eval_dataset = helper.dev_loader   # 加载验证集

    # 加载roberta预训练模型
    roberta_config = BertConfig.from_pretrained('./roberta_pretrain/bert_config.json')
    encoder = BertModel.from_pretrained(args.bert_model, config=roberta_config)

    args.input_dim = roberta_config.hidden_size

    model = BertSupportNet(config=args, encoder=encoder)  # 这里只是bert进行编码

    if args.trained_weight is not None:
        model.load_state_dict(torch.load(args.trained_weight))

    model.to(Config.device)

    # Initialize optimizer and criterions
    lr = args.lr
    t_total = len(Full_Loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = 0.1 * t_total
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    sp_loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    model.train()
    # Training
    global_step = epc = 0
    total_train_loss = [0] * 5
    test_loss_record = []
    VERBOSE_STEP = 1
    while True:
        if epc == args.epochs:  # 5 + 30
            exit(0)
        epc += 1

        Loader = Full_Loader
        Loader.refresh()   # 将数据进行了打乱操作

        if epc > 2:
            train_epoch(Loader, model, predict_during_train=False)
        else:
            train_epoch(Loader, model)
