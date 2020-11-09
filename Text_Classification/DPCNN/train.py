# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2020/11/07 17:08:14
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import os
import gzip
import torch
import random
import pickle
import time
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import precision_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from model import Model
from config import set_args
from apex import amp


class InputFeatures(object):
    def __init__(self, input_ids, bigram, trigram, seq_len, label):
        self.input_ids = input_ids
        self.bigram = bigram
        self.trigram = trigram
        self.seq_len = seq_len
        self.label = label
    
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "input_ids: %s" % (str(self.input_ids))
        s += ", bigram: %s" % (self.bigram)
        s += ", trigram: %s" % (self.trigram)
        s += ", seq_len: %s" % (self.seq_len)
        s += ", label: %d" % (self.label)
        return s


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def evaluate(epoch):
    print("***** Running evaluating *****")
    print("  Num examples = {}".format(len(eval_features)))
    print("  Batch size = {}".format(args.eval_batch_size))

    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_bigram = torch.tensor([f.bigram for f in eval_features], dtype=torch.long)
    eval_trigram = torch.tensor([f.trigram for f in eval_features], dtype=torch.long)
    eval_seq_len = torch.tensor([f.seq_len for f in eval_features], dtype=torch.long)
    eval_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(eval_input_ids, eval_bigram, eval_trigram, eval_seq_len, eval_label)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    eval_loss = 0
    step = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_fnt = CrossEntropyLoss()
    for input_ids, bigram, trigram, seq_len, label in tqdm(eval_dataloader, desc='Evaluation'):
        step += 1
        input_ids = input_ids.cuda()
        seq_len = seq_len.cuda()
        label = label.cuda()
        with torch.no_grad():
            logits = model(input_ids, seq_len, label)
            loss = loss_fnt(logits, label)
        eval_loss += loss.mean().item()  # 统计一个batch的损失 一个累加下去
        labels = label.data.cpu().numpy()
        predic = torch.max(logits.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

    # 损失 召回率 查准率
    eval_loss = eval_loss / step
    eval_accuracy = accuracy_score(labels_all, predict_all)
    eval_recall = recall_score(labels_all, predict_all)
    eval_precision = precision_score(labels_all, predict_all)
    eval_f1_score = f1_score(labels_all, predict_all)
    s = 'epoch:{}, eval_loss: {}, eval_precision: {}, eval_accuracy:{}, eval_recall:{}, eval_f1_score:{}'.format(epoch, eval_loss, eval_precision, eval_accuracy, eval_recall, eval_f1_score)
    print(s)
    s += '\n'
    if not os.path.isfile(args.save_log_file):
        os.mknod(args.save_log_file) 
    with open(args.save_log_file, 'a+') as f:
        f.write(s)
    return eval_loss, eval_accuracy


if __name__ == '__main__':
    args = set_args()
    set_seed(args)  # 设定随机种子
    # device = torch.device('cuda: {}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # 加载训练集
    print('加载训练集:', args.train_features_path)
    with gzip.open(args.train_features_path, 'rb') as f:
        train_features = pickle.load(f)
    print('加载测试集:', args.eval_features_path)
    with gzip.open(args.eval_features_path, 'rb') as f:
        eval_features = pickle.load(f)

    model = Model(args)

    # 模型
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)   # 调整学习率
    
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    if torch.cuda.device_count() > 1:
        args.n_gpu = torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    best_loss = None
    global_step = 0
    if args.do_train:
        print("***** Running training *****")
        print("  Num examples = {}".format(len(train_features)))
        print("  Batch size = {}".format(args.train_batch_size))
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_seq_len = torch.tensor([f.seq_len for f in train_features], dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_seq_len, all_label)
        model.train()
        loss_fnt = CrossEntropyLoss()
        for epoch in range(args.num_train_epochs):
            scheduler.step()   # 学习率衰减
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
            for step, batch in enumerate(train_dataloader):
                start_time = time.time()
                batch = tuple(t.cuda() for t in batch)
                input_ids, seq_len, label = batch
                logits = model(input_ids, seq_len, label)
                loss = loss_fnt(logits, label)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                print('DPCNN-model*****epoch:{}, step:{}, loss:{:10f}, time_cost:{:10f}'.format(epoch, step, loss, time.time()-start_time))
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                # loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    # scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # 验证验证集
            test_loss, test_acc = evaluate(epoch)
            # 验证训练集中四万之后的数据
            model.train()

            if best_loss is None or best_loss > test_loss:
                best_loss = test_loss
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                os.makedirs(args.save_teacher_model, exist_ok=True)
                output_model_file = os.path.join(args.save_teacher_model, "best_pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.save_teacher_model, "epoch{}_ckpt.bin".format(epoch))
            torch.save(model_to_save.state_dict(), output_model_file)

