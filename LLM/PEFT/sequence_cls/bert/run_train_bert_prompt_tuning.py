"""
@file   : run_train_bert_prompt_tuning.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2023-04-20
"""
import os
import torch
import pandas as pd
from torch import nn
from tqdm import tqdm
from config import set_args
from sklearn import metrics
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.models.bert import BertTokenizer
from data_helper import CustomDataset, collate_fn
from transformers import get_linear_schedule_with_warmup
from transformers.models.bert import BertForSequenceClassification
from peft import get_peft_model, PromptTuningConfig, PeftType


def evaluate():
    model.eval()
    eval_targets = []
    eval_predict = []

    for batch in tqdm(val_dataloader):
        if torch.cuda.is_available():
            batch = (t.cuda() for t in batch)
        input_ids, attention_mask, token_type_ids, label_ids = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        eval_targets.extend(label_ids.cpu().detach().numpy().tolist())
        eval_predict.extend(torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist())
    eval_accuracy = metrics.accuracy_score(eval_targets, eval_predict)
    return eval_accuracy


if __name__ == '__main__':
    args = set_args()
    args.output_dir = './prompt_tuning'
    os.makedirs(args.output_dir, exist_ok=True)

    peft_type = PeftType.PROMPT_TUNING
    peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10, peft_type=peft_type)
    lr = 1e-3

    # 不同的模型 padding有不同
    model_name_or_path = 'bert'
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, padding_side=padding_side)

    if getattr(tokenizer, "pad_token_id") is None:  # 没有padding token的 都按eos填充
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 准备数据
    train_df = pd.read_csv(args.train_data_path)
    train_dataset = CustomDataset(dataframe=train_df, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    val_df = pd.read_csv(args.val_data_path)
    val_dataset = CustomDataset(dataframe=val_df, tokenizer=tokenizer)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=args.val_batch_size,
                                collate_fn=collate_fn)

    # 准备模型
    model = BertForSequenceClassification.from_pretrained(args.pretrained_model_path, return_dict=True, num_labels=2)
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())
    # trainable params: 296450 || all params: 102564098 || trainable: 0.28903876286222496

    optimizer = AdamW(params=model.parameters(), lr=lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,  # 0.06*(len(train_dataloader) * num_epochs),
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    if torch.cuda.is_available():
        model.cuda()

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            input_ids, attention_mask, token_type_ids, label_ids = batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = output.logits
            loss = loss_func(logits, label_ids)
            print('epoch:{}, step:{}, loss:{}'.format(epoch, step, loss))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 保存模型与参数
            peft_config.save_pretrained(args.output_dir)
            model.save_pretrained(args.output_dir)
            exit()

        # 一轮跑完  做个模型验证
        eval_acc = evaluate()