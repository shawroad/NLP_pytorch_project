"""
@file   : run_pretrain.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-14
"""
import os
import torch
from torch import nn
from config import set_args
from torch.utils.data import DataLoader
from data_helper import RobertaDataSet, collate_fn
from transformers.models.bert import BertConfig, BertForPreTraining, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter


def get_metric_acc(logits, target):
    # print(logits.size())    # torch.Size([172, 50000])
    # print(target.size())    # torch.Size([172])
    prob, pred_label = torch.max(logits, dim=-1)
    fenzi, fenmu = 0, 0
    for p_lab, t_lab in zip(pred_label, target):
        if t_lab != 0:
            fenmu += 1
            if t_lab == p_lab:
                fenzi += 1
    if fenmu == 0:
        return 1
    acc = fenzi / fenmu
    return acc


def calc_loss(logits, labels):
    # 算了所有预测的损失
    logits = logits.view(-1, tokenizer.vocab_size)
    labels = labels.view(-1)
    loss = loss_fct(logits, labels)
    return loss


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_weight)

    train_dataset = RobertaDataSet(args.train_data_path, tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)

    config = BertConfig.from_pretrained(args.pretrain_weight)

    model = BertForPreTraining(config=config)
    model_weight_path = os.path.join(args.pretrain_weight, 'pytorch_model.bin')
    model.load_state_dict(torch.load(model_weight_path), strict=False)

    loss_fct = nn.CrossEntropyLoss(ignore_index=0)
    if torch.cuda.is_available():
        model.cuda()
        loss_fct = loss_fct.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_train_optimization_steps * args.weight_decay_rate,
                                                num_training_steps=num_train_optimization_steps)
    logger = SummaryWriter(log_dir="log/log")
    global_step = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            input_ids, attention_mask, segment_ids, lm_mask_labels = batch
            outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
            # print(outputs['prediction_logits'].size())   # torch.Size([2, 256, 21128])
            # print(outputs['seq_relationship_logits'].size())   # torch.Size([2, 2])  # 可以不要 因为我们只训练mask任务

            prediction_logits = outputs['prediction_logits']
            loss = calc_loss(prediction_logits, lm_mask_labels)

            acc = get_metric_acc(logits=prediction_logits.view(-1, tokenizer.vocab_size), target=lm_mask_labels.view(-1))

            if global_step % 100 == 0:
                # 每隔100步  保存一下准确率和loss
                logger.add_scalar("train loss", loss.item(), global_step=global_step)
                logger.add_scalar("train accuracy", acc, global_step=global_step)

            loss.backward()
            print("epoch:{}, step:{}/{}, Loss:{:10f}, Accuracy:{:10f}".format(
                epoch, step, len(train_dataloader), loss, acc)
            )
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step % args.save_checkponint_steps == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(args.output_dir, 'pytorch_model_epoch{}.bin'.format(global_step))
                torch.save(model_to_save.state_dict(), output_model_file)

                # save config
                output_config_file = os.path.join(args.output_dir, "config.json")
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())

                # save vocab
                tokenizer.save_vocabulary(args.output_dir)
