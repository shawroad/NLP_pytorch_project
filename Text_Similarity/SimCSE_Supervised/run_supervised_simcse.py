"""
@file   : run_supervised_simcse.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-04
"""
import os
import torch
import numpy as np
import scipy.stats
from model import Model
from config import set_args
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data_helper import load_data, MyDataset, TestDataset, load_STS_data


def compute_loss(y_pred, lamda=0.05):
    # 以下pdb的时候 batch_size=3
    row = torch.arange(0, y_pred.shape[0], 3)    # tensor([0, 3, 6])
    col = torch.arange(y_pred.shape[0])
    col = torch.where(col % 3 != 0)[0]    # tensor([1, 2, 4, 5, 7, 8])
    y_true = torch.arange(0, len(col), 2)   # tensor([0, 2, 4])
    # y_pred.size(): torch.Size([9, 768])
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    # print(similarities.size())   # torch.Size([9, 9])

    # torch自带的快速计算相似度矩阵的方法
    similarities = torch.index_select(similarities, 0, row)
    similarities = torch.index_select(similarities, 1, col)
    # 屏蔽对角矩阵即自身相等的loss
    similarities = similarities / lamda
    # 论文中除以 temperature 超参 0.05
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(dataloader, model, optimizer):
    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].view(len(batch['input_ids']) * 3, -1)
            attention_mask = batch['attention_mask'].view(len(batch['attention_mask']) * 3, -1)
            token_type_ids = batch['token_type_ids'].view(len(batch['token_type_ids']) * 3, -1)

            if torch.cuda.is_available():
                input_ids, attention_mask, token_type_ids = input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda()

            pred = model(input_ids, attention_mask, token_type_ids)
            # print(pred.size())   # batch_size, hidden_size

            loss = compute_loss(pred)
            print('epoch:{}, step:{}, loss:{:10f}'.format(epoch, step, loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        corrcoef = evaluate(deving_data, model)
        model.train()
        ss = 'epoch:{}, spearman:{}'.format(epoch, corrcoef)
        print(ss)
        with open('log.txt', 'a+', encoding='utf8') as f:
            ss += '\n'
            f.write(ss)
        save_model_path = os.path.join(args.output_dir, 'Epoch-{}.bin'.format(epoch))
        torch.save(model.state_dict(), save_model_path)


def evaluate(test_data, model):
    target_idxs, source_idxs, label_list = test_data.get_data()
    with torch.no_grad():
        target_input_ids = target_idxs['input_ids']
        target_attention_mask = target_idxs['attention_mask']
        target_token_type_ids = target_idxs['token_type_ids']
        if torch.cuda.is_available():
            target_input_ids, target_attention_mask, target_token_type_ids = target_input_ids.cuda(), target_attention_mask.cuda(), target_token_type_ids.cuda()
        traget_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)

        source_input_ids = source_idxs['input_ids']
        source_attention_mask = source_idxs['attention_mask']
        source_token_type_ids = source_idxs['token_type_ids']
        if torch.cuda.is_available():
            source_input_ids, source_attention_mask, source_token_type_ids = source_input_ids.cuda(), source_attention_mask.cuda(), source_token_type_ids.cuda()
        source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)

        similarity_list = F.cosine_similarity(traget_pred, source_pred)
        similarity_list = similarity_list.cpu().numpy()
        label_list = np.array(label_list)
        corrcoef = scipy.stats.spearmanr(label_list, similarity_list).correlation
    return corrcoef


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    # 加载数据集   # 关系为: contradiction、neutral、entailment
    snil_data = load_data(args.train_data_path)    # [{'sentence1': 'xxx', 'sentence2': 'xxx', 'gold_label':'xxx'}, ...]
    dev_data = load_STS_data("./data/STS-B/cnsd-sts-dev.txt")
    np.random.shuffle(snil_data)

    training_data = MyDataset(snil_data, tokenizer, args.maxlen)

    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    deving_data = TestDataset(dev_data, tokenizer, args.maxlen)

    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train(train_dataloader, model, optimizer)