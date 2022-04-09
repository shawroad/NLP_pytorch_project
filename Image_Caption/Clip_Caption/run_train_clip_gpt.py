"""
@file   : run_train_clip_gpt.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-30
"""
import os
import torch
from tqdm import tqdm
from config import set_args
from os.path import join
import torch.nn.functional as F
from model import ClipCaptionModel
from data_helper import ClipCapDataset
from torch.utils.data import Dataset, DataLoader
from transformers.models.bert import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup


def evaluate(args, model, dataloader):
    model.eval()
    eval_loss = 0.0  #
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            clip_embeds, caption_ids, mask = batch
            logits = model(clip_embeds, caption_ids, mask)

            # 计算loss
            shift_logits = logits[..., args.prefix_len - 1:-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = caption_ids.view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)
            loss = loss.mean()  # 对多卡的loss取平均
            eval_loss += loss
    return eval_loss


def main():
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(args.gpt2_path)

    # 1. 准备数据集
    all_dataset = ClipCapDataset(args.prefix_len, tokenizer, args.max_len)
    # 切分训练 验证集
    train_dataset, dev_dataset = torch.utils.data.random_split(all_dataset, [len(all_dataset) - args.dev_size, args.dev_size])
    print('训练集大小:', len(train_dataset))
    print('验证集大小:', len(dev_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)   # num_wokers
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    total_step = len(train_dataloader) * args.epochs

    model = ClipCaptionModel()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_step)

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            clip_embeds, caption_ids, mask = batch

            logits = model(clip_embeds, caption_ids, mask)

            # 计算loss
            shift_logits = logits[..., args.prefix_len - 1:-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = caption_ids.view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)
            print('epoch:{}, step:{}, loss:{}'.format(epoch, step, loss))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % args.eval_step == 0:
                dev_loss = evaluate(args, model, dev_dataloader)
                ss = 'epoch:{}, step:{}, loss:{}'.format(epoch, step, dev_loss)
                logs_path = os.path.join(args.output_dir, 'logs.txt')
                with open(logs_path, 'a+') as f:
                    ss += '\n'
                    f.write(ss)

                output_model_file = os.path.join(args.output_dir, "base_model_epoch{}_step{}.bin".format(epoch, step))
                torch.save(model.state_dict(), output_model_file)
                model.train()
        output_model_file = os.path.join(args.output_dir, "base_model_epoch{}.bin".format(epoch))
        torch.save(model.state_dict(), output_model_file)
        model.train()


if __name__ == '__main__':
    args = set_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main()
    
