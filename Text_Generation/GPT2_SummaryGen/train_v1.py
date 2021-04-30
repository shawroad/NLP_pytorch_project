# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 13:56
# @Author  : xiaolu
# @FileName: train.py
# @Software: PyCharm

import torch
import os
from transformers import BertTokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel
from torch.utils.data import DataLoader
from transformers import modeling_gpt2
from datetime import datetime
from os.path import join
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert.optimization import BertAdam
import numpy as np
from dataset import MyDataset
from config import Config

PAD = '[PAD]'
pad_id = 0


def calculate_loss_and_accuracy(outputs, labels):
    """
    计算非pad_id的平均loss和准确率
    :param outputs:
    :param labels:
    :param device:
    :return:
    """
    logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
    # print(logits.size())   # torch.Size([2, 1024, 13317])

    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 错位预测 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits[..., :-1, :].contiguous()
    # print(shift_logits.size())    # torch.Size([2, 1023, 13317])

    shift_labels = labels[..., 1:].contiguous().to(Config.device)
    # print(shift_labels.size())   # torch.Size([2, 1023])

    # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在vocab中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy


def collate_fn(batch):
    '''
    batch for padding 计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    '''
    global pad_id
    input_ids = []
    btc_size = len(batch)
    max_input_len = 0

    # 计算该batch中input的最大长度
    for bat_idx in range(btc_size):
        if max_input_len < len(batch[bat_idx]):
            max_input_len = len(batch[bat_idx])

    # 使用pad_id对小于max_input_len的input_id进行补全
    for bat_idx in range(btc_size):
        input_len = len(batch[bat_idx])
        input_ids.append(batch[bat_idx])
        input_ids[bat_idx].extend([pad_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def train(model, train_list, test_list):
    train_dataset = MyDataset(train_list)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=collate_fn, drop_last=True)
    model.train()

    # 计算所有epoch进行参数优化的总步数total_steps
    total_steps = int(train_dataset.__len__() * Config.epochs / Config.batch_size / Config.gradient_accumulation)
    print("total train step num: {}".format(total_steps))

    optimizer = BertAdam(model.parameters(), lr=Config.lr, warmup=0.05, t_total=total_steps)
    print('start training...')
    # 开始训练
    for epoch in range(Config.epochs):
        epoch_start_time = datetime.now()
        for batch_idx, input_ids in enumerate(train_dataloader):
            # 注意：GPT2模型的forward()函数，是对于给定的context，生成一个token，而不是生成一串token
            # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token
            input_ids = input_ids.to(Config.device)

            outputs = model.forward(input_ids=input_ids)

            loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            print('epoch:{}, step:{}, loss: {:6f}, accuracy:{:6f}'.format(epoch + 1, batch_idx + 1, loss, accuracy))

        average_acc, average_loss = evaluate(model, test_list)
        res = "VALID epoch:{}, loss {:6f},  acc {:6f}".format(epoch, average_loss, average_acc)
        print(res)
        res += '\n'
        with open('log.txt', 'a+') as f:
            f.write(res)
        # 一个epoch跑完保存一下模型
        model_path = join(Config.model_output_path, 'model_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_path)
        epoch_finish_time = datetime.now()
        print('跑完一个epoch花费时间为: {}'.format(epoch_finish_time - epoch_start_time))


def evaluate(model, test_list):
    '''
    验证
    :param model:
    :param test_list:
    :return:
    '''
    model.eval()
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2,
                                 collate_fn=collate_fn, drop_last=True)
    print("starting evaluating")
    acc = []
    losses = []
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(test_dataloader):
            input_ids = input_ids.to(Config.device)
            outputs = model.forward(input_ids=input_ids)
            loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids)
            acc.append(accuracy.item())
            losses.append(loss.item())

    average_acc = np.mean(acc)
    average_loss = np.mean(losses)
    return average_acc, average_loss


def create_model(vocab_size):
    '''
    创建模型
    :return:
    '''
    if Config.pretrained_model:
        # 若有预训练模型 则加载
        model = GPT2LMHeadModel.from_pretrained(Config.pretrained_model)
    else:
        # 若没有预训练模型  则创建模型  从头训练起
        model_config = modeling_gpt2.GPT2Config.from_json_file(Config.gpt2_config)
        model = GPT2LMHeadModel(config=model_config)

    # 根据tokenizer的vocabulary调整GPT2模型的vocab的大小
    model.resize_token_embeddings(vocab_size)
    return model, model.config.to_dict().get("n_ctx")  # 输入长度


def main():
    # 初始化tokenizer
    tokenizer = BertTokenizer(vocab_file='./gpt2_model/vocab_small.txt')
    # tokenizer的字典大小
    vocab_size = len(tokenizer)
    # print(vocab_size)  # 13317

    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(PAD)
    # print(pad_id)  # 0

    # 创建模型输出路径
    if not os.path.exists(Config.model_output_path):
        os.mkdir(Config.model_output_path)

    # 加载GPT2模型
    model, n_ctx = create_model(vocab_size)
    model.to(Config.device)

    # 统计模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('模型的总参数量: {}'.format(num_parameters))   # 81894144

    # 加载数据
    with open(Config.train_tokenized_path, 'r', encoding='utf8') as f:
        data = f.read()
    data_list = data.split("\n")

    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=1)  # 切分训练集和测试集
    # 开始训练   边训练边测试
    # train(model, train_list)
    train(model, train_list, test_list)

    # 测试模型
    evaluate(model, test_list)


if __name__ == '__main__':
    main()
