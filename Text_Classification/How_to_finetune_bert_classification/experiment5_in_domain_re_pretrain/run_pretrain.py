"""
@file   : run_pretrain.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-05
"""
import argparse
import os
import torch
import torch.nn as nn
import random
import numpy as np
from data_loader import MyDataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForPreTraining
from transformers.models.bert import BertTokenizer, BertConfig
from pdb import set_trace


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--max_predictions_per_seq', default=20, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_warmup_steps', default=10000, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--bert_pretrain', default='../bert_pretrain', type=str, help='bert的预训练模型')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='模型输出')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained('../bert_pretrain/vocab.txt')
    train_data_path = './process_data.json'
    dataset = MyDataset(train_data_path, tokenizer)
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    bert_config = BertConfig.from_pretrained('../bert_pretrain/config.json')

    model = BertForPreTraining.from_pretrained(args.bert_pretrain)

    if torch.cuda.is_available():
        model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=args.num_warmup_steps,
                    num_training_steps=len(train_dataloader) * args.num_epochs)
    total_step = len(train_dataloader)

    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(args.num_epochs):
        model.train()
        model.zero_grad()
        for step, (cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_masked_lm_labels, cur_next_sentence_labels) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                cur_input_ids, cur_attention_mask = cur_input_ids.cuda(), cur_attention_mask.cuda()
                cur_token_type_ids, cur_masked_lm_labels = cur_token_type_ids.cuda(), cur_masked_lm_labels.cuda()
                cur_next_sentence_labels = cur_next_sentence_labels.cuda()

            outputs = model(input_ids=cur_input_ids, token_type_ids=cur_token_type_ids, attention_mask=cur_attention_mask)
            prediction_scores = outputs[0]
            seq_relationship_score = outputs[1]
            # print(prediction_scores.size())    # torch.Size([32, 256, 30522])
            # print(seq_relationship_score.size())   # torch.Size([32, 2])

            masked_lm_loss = loss_fct(prediction_scores.view(-1, bert_config.vocab_size), cur_masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), cur_next_sentence_labels.view(-1))
            loss = masked_lm_loss + next_sentence_loss
            print('epoch:{}, step:{}, mask_lm_loss:{:6f}, next_sentence_loss:{:6f}, sum_loss:{:6f}'.format(
                epoch, step, masked_lm_loss, next_sentence_loss, loss))
            if args.gradient_accumulation_step > 1:
                loss /= args.gradient_accumulation_step

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if (step + 1) % args.gradient_accumulation_step == 0:
                # nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        output_path = args.output_dir + '/' + 'epoch_{}'.format(epoch)
        model.save_pretrained(output_path)
        # 再训练之后，可以采用再训练完后的代码进行微调哦!

