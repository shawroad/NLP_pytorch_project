# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 17:14
# @Author  : xiaolu
# @FileName: run_pretrain.py
# @Software: PyCharm
import torch
import time
import os
from transformers import BertTokenizer
from torch.utils.data.dataloader import DataLoader
from transformers import BertConfig, BertForPreTraining
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from common import AverageMeter
from custom_metrics import LMAccuracy
from data_loader import Data_pretrain
from config import Config


if __name__ == '__main__':
    #  training_path, file_id, tokenizer, data_name, reduce_memory=False
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain/vocab.txt')
    train_data_path = './process_data0.json'
    txt = Data_pretrain(train_data_path, tokenizer)
    data_iter = DataLoader(txt, shuffle=True, batch_size=2)
    bert_config = BertConfig.from_pretrained(Config.config_path)
    model = BertForPreTraining(config=bert_config)

    model.to(Config.device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    num_train_optimization_steps = len(data_iter) * Config.epochs
    warmup_steps = int(num_train_optimization_steps * Config.warmup_proportion)

    optimizer = AdamW(params=optimizer_grouped_parameters, lr=Config.learning_rate, eps=Config.adam_epsilon)
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
    for epc in range(Config.epochs):
        for step, batch in enumerate(data_iter):
            batch = tuple(t.to(Config.device) for t in batch)
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

            if Config.gradient_accumulation_steps > 1:
                loss = loss / Config.gradient_accumulation_steps
            loss.backward()

            nb_tr_steps += 1
            tr_mask_acc.update(mask_metric.value(), n=input_ids.size(0))
            tr_sop_acc.update(sop_metric.value(), n=input_ids.size(0))
            tr_loss.update(loss.item(), n=1)
            tr_mask_loss.update(masked_lm_loss.item(), n=1)
            tr_sop_loss.update(next_sentence_loss.item(), n=1)

            if (step + 1) % Config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % Config.num_save_steps == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(Config.output_dir, 'pytorch_model_epoch{}.bin'.format(global_step))
                torch.save(model_to_save.state_dict(), output_model_file)

                # save config
                output_config_file = Config.output_dir + "config.json"
                with open(str(output_config_file), 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                # save vocab
                tokenizer.save_vocabulary(Config.output_dir)


