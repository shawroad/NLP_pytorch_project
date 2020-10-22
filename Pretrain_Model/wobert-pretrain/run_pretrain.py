# -*- encoding: utf-8 -*-
'''
@File    :   run_pretrain.py
@Time    :   2020/10/22 08:35:01
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import torch
import time
import os
import argparse
import random
import numpy as np
from torch import nn
from transformers import BertTokenizer
from torch.utils.data.dataloader import DataLoader
from transformers import BertConfig, BertForPreTraining
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from common import AverageMeter
from custom_metrics import LMAccuracy
from data_loader import Data_pretrain


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_model", default='./wobert_pretrain/pytorch_model.bin', type=str, help='pretrain_model')
    parser.add_argument("--model_config", default='./wobert_pretrain/bert_config.json', type=str, help='pretrain_model_config')
    parser.add_argument("--vocab_file", default='./wobert_pretrain/vocab.txt')

    parser.add_argument("--train_data_path", default='./data/processed_data0.json', type=str, help='data with training')
    parser.add_argument('--batch_size', type=int, default=32, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=20, help="random seed for initialization")
    parser.add_argument('--learning_rate', type=float, default=0.000176, help="random seed for initialization")
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help="random seed for initialization")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help="random seed for initialization")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="random seed for initialization")
    
    parser.add_argument('--n_gpu', type=int, default=1, help="random seed for initialization")
    parser.add_argument('--seed', type=int, default=43, help="random seed for initialization")
    parser.add_argument("--output_dir", default='./retrain_model', type=str)
    
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.vocab_file)

    txt = Data_pretrain(args.train_data_path, tokenizer)
    data_iter = DataLoader(txt, shuffle=True, batch_size=args.batch_size)
    bert_config = BertConfig.from_pretrained(args.model_config)
    model = BertForPreTraining(config=bert_config)

    # 模型
    if torch.cuda.is_available():
        model.cuda()
    
    if torch.cuda.device_count() > 1:
        args.n_gpu = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    set_seed(args)
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    num_train_optimization_steps = len(data_iter) * args.epochs
    warmup_steps = int(num_train_optimization_steps * args.warmup_proportion)

    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)
    loss_fct = CrossEntropyLoss(ignore_index=-1)

    mask_metric = LMAccuracy()
    sop_metric = LMAccuracy()
    tr_mask_acc = AverageMeter()
    tr_sop_acc = AverageMeter()
    tr_loss = AverageMeter()
    tr_mask_loss = AverageMeter()
    tr_sop_loss = AverageMeter()

    train_logs = {}
    nb_tr_steps = 0
    global_step = 0
    start_time = time.time()
    for epc in range(args.epochs):
        for step, batch in enumerate(data_iter):
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
            outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            
            prediction_scores = outputs[0]
            seq_relationship_score = outputs[1]

            masked_lm_loss = loss_fct(prediction_scores.view(-1, bert_config.vocab_size), lm_label_ids.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), is_next.view(-1))
            loss = masked_lm_loss + next_sentence_loss
            print('epoch:{}, step:{}, mask_lm_loss:{:6f}, next_sentence_loss:{:6f}, sum_loss:{:6f}'.format(
                epc, step, masked_lm_loss, next_sentence_loss, loss))

            mask_metric(logits=prediction_scores.view(-1, bert_config.vocab_size), target=lm_label_ids.view(-1))
            sop_metric(logits=seq_relationship_score.view(-1, 2), target=is_next.view(-1))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            nb_tr_steps += 1
            tr_mask_acc.update(mask_metric.value(), n=input_ids.size(0))
            tr_sop_acc.update(sop_metric.value(), n=input_ids.size(0))
            tr_loss.update(loss.item(), n=1)
            tr_mask_loss.update(masked_lm_loss.item(), n=1)
            tr_sop_loss.update(next_sentence_loss.item(), n=1)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        model_to_save = model.module if hasattr(model, 'module') else model
        os.makedirs(args.output_dir, exist_ok=True)
        output_model_file = os.path.join(args.output_dir, 'pytorch_model_epoch{}.bin'.format(global_step))
        torch.save(model_to_save.state_dict(), output_model_file)
        # save config
        output_config_file = args.output_dir + "config.json"
        with open(str(output_config_file), 'w') as f:
            f.write(model_to_save.config.to_json_string())
        # save vocab
        tokenizer.save_vocabulary(args.output_dir)