"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-07
"""
import os
import time
import torch
import torch.nn as nn
from model import Model
from tqdm import tqdm
from config import set_args
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from data_helper import LCQMC_Dataset, load_embeddings


def correct_predictions(output_probabilities, targets):
    '''
    计算正确样本的个数
    :param output_probabilities:
    :param targets:
    :return:
    '''
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def evaluate():
    model.eval()
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    with torch.no_grad():
        for (q, q_len, h, h_len, label) in tqdm(dev_loader):
            batch_start = time.time()
            q1, q2, q1_len, q2_len, label = q, h, q_len, h_len, label
            if torch.cuda.is_available():
                q1, q2, q1_len, q2_len, label = q.cuda(), h.cuda(), q_len.cuda(), h_len.cuda(), label.cuda()
            _, probs = model(q1, q1_len, q2, q2_len)
            accuracy += correct_predictions(probs, label)
            batch_time += time.time() - batch_start
            all_prob.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(label)

    batch_time /= len(dev_loader)
    total_time = time.time() - time_start
    accuracy /= (len(dev_loader.dataset))
    return batch_time, total_time, accuracy, roc_auc_score(all_labels, all_prob)


def train():
    criterion = nn.CrossEntropyLoss()

    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)
    best_auc_score = 0.0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            start_time = time.time()
            optimizer.zero_grad()
            p, p_len, h, h_len, label = batch
            if torch.cuda.is_available():
                p, p_len, h, h_len, label = p.cuda(), p_len.cuda(), h.cuda(), h_len.cuda(), label.cuda()

            logits, probs = model(p, p_len, h, h_len)
            loss = criterion(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # 计算当前batch的准确率
            b_acc = correct_predictions(probs, label) / probs.size(0)
            print('epoch:{}, step:{}, loss:{:.4f}, accuracy:{:.4f}, time:{:.4f}s'.format(epoch, step, loss, b_acc, time.time() - start_time))
            exit()


        batch_time, total_time, eval_accuracy, eval_auc = evaluate()
        if eval_auc > best_auc_score:
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            os.makedirs(args.save_model, exist_ok=True)
            output_model_file = os.path.join(args.save_model, "best_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            best_auc_score = eval_auc


if __name__ == "__main__":
    args = set_args()

    vocab_file = '../data/vocab.txt'
    train_file='../data/LCQMC/LCQMC.train.data'
    valid_file = '../data/LCQMC/LCQMC.valid.data'
    embeddings_file = '../data/token_vec_300.bin'

    print('加载训练集ing...')
    train_data = LCQMC_Dataset(train_file, vocab_file, args.max_char_len)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)

    print('加载验证集ing...')
    dev_data = LCQMC_Dataset(valid_file, vocab_file, args.max_char_len)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=args.dev_batch_size)

    print('加载词向量ing...')
    embeddings = load_embeddings(embeddings_file)

    model = Model(embeddings)
    if torch.cuda.is_available():
        model.cuda()

    train()





