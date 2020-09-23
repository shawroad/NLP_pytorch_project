# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 9:33
# @Author  : xiaolu
# @FileName: train.py
# @Software: PyCharm
import os
import gzip
import torch
import random
import pickle
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from model import Model
from config import set_args


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "  input_ids: %s" % (str(self.input_ids))
        s += ", input_mask: %s" % (self.input_mask)
        s += ", segment_ids: %s" % (self.segment_ids)
        s += ", label: %s" % (self.label)
        return s


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def evaluate(epoch):
    print("***** Running evaluating *****")
    print("  Num examples = {}".format(len(eval_features)))
    print("  Batch size = {}".format(args.eval_batch_size))
    question_input_ids = torch.tensor([f[0].input_ids for f in eval_features], dtype=torch.long)
    question_input_mask = torch.tensor([f[0].input_mask for f in eval_features], dtype=torch.long)
    question_segment_ids = torch.tensor([f[0].segment_ids for f in eval_features], dtype=torch.long)
    label = torch.tensor([f[0].label for f in eval_features], dtype=torch.long)

    context_input_ids = torch.tensor([f[1].input_ids for f in eval_features], dtype=torch.long)
    context_input_mask = torch.tensor([f[1].input_mask for f in eval_features], dtype=torch.long)
    context_segment_ids = torch.tensor([f[1].segment_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(question_input_ids, question_input_mask, question_segment_ids,
                               context_input_ids, context_input_mask, context_segment_ids, label)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss = 0
    step = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for ques_input_ids, ques_input_mask, ques_segment_ids, context_input_ids, context_input_mask, context_segment_ids, labels_ids in tqdm(eval_dataloader, desc='Evaluation'):
        step += 1
        ques_input_ids = ques_input_ids.to(device)
        ques_input_mask = ques_input_mask.to(device)
        ques_segment_ids = ques_segment_ids.to(device)

        context_input_ids = context_input_ids.to(device)
        context_input_mask = context_input_mask.to(device)
        context_segment_ids = context_segment_ids.to(device)

        label_ids = labels_ids.to(device)
        with torch.no_grad():
            loss, logits = model(ques_input_ids=ques_input_ids, ques_input_mask=ques_input_mask,
                                 ques_segment_ids=ques_segment_ids, context_input_ids=context_input_ids,
                                 context_input_mask=context_input_mask, context_segment_ids=context_segment_ids,
                                 labels=label_ids)

        eval_loss += loss.mean().item()  # 统计一个batch的损失 一个累加下去

        labels = label_ids.data.cpu().numpy()
        predic = torch.max(logits.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

    # 损失 召回率 查准率
    eval_loss = eval_loss / step
    eval_accuracy = accuracy_score(labels_all, predict_all)
    eval_recall = recall_score(labels_all, predict_all)
    s = 'epoch:{}, eval_loss: {}, eval_accuracy:{}, eval_recall:{}'.format(epoch, eval_loss, eval_accuracy, eval_recall)
    print(s)
    s += '\n'
    with open('result_eval.txt', 'a+') as f:
        f.write(s)
    return eval_loss, eval_accuracy


if __name__ == '__main__':
    args = set_args()
    set_seed(args)  # 设定随机种子

    device = torch.device('cuda: {}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # 加载训练集
    with gzip.open(args.train_features_path, 'rb') as f:
        train_features = pickle.load(f)

    with gzip.open(args.eval_features_path, 'rb') as f:
        eval_features = pickle.load(f)

        # Prepare Optimizer
    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # 模型
    model = Model()
    tokenizer = BertTokenizer.from_pretrained(args.vocab_file)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = 0.05 * num_train_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    best_loss = None
    global_step = 0
    model.to(device)

    if args.do_train:
        print("***** Running training *****")
        print("  Num examples = {}".format(len(train_features)))
        print("  Batch size = {}".format(args.train_batch_size))
        question_input_ids = torch.tensor([f[0].input_ids for f in train_features], dtype=torch.long)
        question_input_mask = torch.tensor([f[0].input_mask for f in train_features], dtype=torch.long)
        question_segment_ids = torch.tensor([f[0].segment_ids for f in train_features], dtype=torch.long)
        label = torch.tensor([f[0].label for f in train_features], dtype=torch.long)

        context_input_ids = torch.tensor([f[1].input_ids for f in train_features], dtype=torch.long)
        context_input_mask = torch.tensor([f[1].input_mask for f in train_features], dtype=torch.long)
        context_segment_ids = torch.tensor([f[1].segment_ids for f in train_features], dtype=torch.long)
        train_data = TensorDataset(question_input_ids, question_input_mask, question_segment_ids,
                                   context_input_ids, context_input_mask, context_segment_ids, label)

        model.train()
        for epoch in range(args.num_train_epochs):
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
            for step, batch in enumerate(train_dataloader):

                start_time = time.time()
                batch = tuple(t.to(device) for t in batch)
                ques_input_ids, ques_input_mask, ques_segment_ids, context_input_ids, context_input_mask, context_segment_ids, labels_ids = batch
                # print(ques_input_ids.size())     # torch.Size([2, 61])
                # print(ques_input_mask.size())    # torch.Size([2, 61])
                # print(ques_segment_ids.size())   # torch.Size([2, 61])
                # print('*'*100)
                # print(context_input_ids.size())    # torch.Size([2, 441])
                # print(context_input_mask.size())    # torch.Size([2, 441])
                # print(context_segment_ids.size())    # torch.Size([2, 441])
                # print(labels_ids.size())    # torch.Size([2])
                loss, logits = model(ques_input_ids=ques_input_ids, ques_input_mask=ques_input_mask,
                                     ques_segment_ids=ques_segment_ids, context_input_ids=context_input_ids,
                                     context_input_mask=context_input_mask, context_segment_ids=context_segment_ids,
                                     labels=labels_ids)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                print('epoch:{}, step:{}, loss:{:10f}, time_cost:{:10f}'.format(epoch, step, loss, time.time()-start_time))
                loss.backward()

                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # 一个epoch跑完 然后进行验证
            os.makedirs(args.output_dir, exist_ok=True)
            output_prediction_file = os.path.join(args.output_dir, "epoch{}_prediction.csv".format(epoch))

            # 验证验证集
            test_loss, test_acc = evaluate(epoch)
            # 验证训练集中四万之后的数据
            model.train()

            if best_loss is None or best_loss > test_loss:
                best_loss = test_loss
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                os.makedirs(args.ckpt_dir, exist_ok=True)
                output_model_file = os.path.join(args.ckpt_dir, "best_pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.ckpt_dir, "epoch{}_ckpt.bin".format(epoch))
            torch.save(model_to_save.state_dict(), output_model_file)

