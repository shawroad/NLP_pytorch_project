"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-04-30
"""
import torch
import random
import os
from tqdm import tqdm
import numpy as np
from config import set_args
from model import GPT2LMHeadModel
from transformers import BertTokenizer, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from data_process import GPT2NewsTitleDataSet, collate_func
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def train(model, train_data, test_data, args):
    tb_write = SummaryWriter()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps参数无效，必须大于等于1")

    # 计算真实的训练batch_size大小
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size, collate_fn=collate_func)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    print("总训练步数为:{}".format(total_steps))

    if torch.cuda.is_available():
        model.cuda()

    # 获取模型所有参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)

    # 将模型调至训练状态
    model.train()

    title_id = train_data.title_id
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    # 开始训练模型
    torch.cuda.empty_cache()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            input_ids = batch["input_ids"]
            token_type_ids = batch["token_type_ids"]

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()

            # 获取训练结果
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids, title_id=title_id)
            loss = outputs[0]

            tr_loss += loss.item()
            # 将损失值放到Iter中，方便观察
            print('Epoch:{}, Step:{}, Loss:{}'.format(epoch, step, loss))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # 损失进行回传
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 如果步数整除logging_steps，则记录学习率和训练集损失值
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_write.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_write.add_scalar("train_loss", (tr_loss-logging_loss) /
                                        (args.logging_steps * args.gradient_accumulation_steps), global_step)
                    logging_loss = tr_loss

                # 如果步数整除eval_steps，则进行模型测试，记录测试集的损失
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, test_data, args)
                    tb_write.add_scalar("test_loss", eval_loss, global_step)
                    model.train()

        # 每个epoch进行完，则保存模型
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        # 清空cuda缓存
        torch.cuda.empty_cache()


def evaluate(model, test_data, args):
    '''
    模型验证
    :param model:
    :param test_data:
    :param args:
    :return:
    '''
    # 构造测试集的DataLoader
    test_sampler = SequentialSampler(test_data)
    test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size, collate_fn=collate_func)
    title_id = test_data.title_id
    total_loss, total = 0.0, 0.0
    # 进行测试
    for step, batch in tqdm(enumerate(test_data_loader)):
        # 模型设为eval
        model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"]
            token_type_ids = batch["token_type_ids"]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()

            # 获取预测结果
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids, title_id=title_id)
            loss = outputs[0]
            loss = loss.item()

            # 对loss进行累加
            total_loss += loss * len(batch["input_ids"])
            total += len(batch["input_ids"])
    # 计算最终测试集的loss结果
    test_loss = total_loss / total
    return test_loss


def main():
    # 设置模型训练参数
    args = set_args()

    # 设置随机种子，方便模型复现
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    # 加载模型的config
    model_config = GPT2Config.from_json_file(args.config_path)

    # 实例化GPT2LMHeadModel模型，这里我们没有加载预训练好的模型，而是直接从头开始训练。
    if args.pretrained_model_path:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
    else:
        # 如果没有指定的预训练模型，则初始化模型
        model = GPT2LMHeadModel(config=model_config)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)

    # 将[space]作为一个分割整体，例如："我爱[Space]中国。"，使用原始tokenizer分词结果为"['我', '爱', '[', 'Space', ']', '中', '国', '。']";
    # 增加分割符号后的结果为"['我', '爱', '[Space]', '中', '国', '。']"
    tokenizer.add_tokens("[Space]", special_tokens=True)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # 加载训练数据和测试数据
    train_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "train", args.train_file_path)
    test_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "test", args.test_file_path)
    # 开始训练
    train(model, train_data, test_data, args)


if __name__ == '__main__':
    main()
