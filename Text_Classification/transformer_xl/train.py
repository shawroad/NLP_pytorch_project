# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2020/10/30 19:54:09
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
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from model import MemoryTransformerXL
from config import set_args
from pdb import set_trace
from apex import amp


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def evaluate(epoch):
    print("***** Running evaluating *****")
    print("  Num examples = {}".format(len(eval_features)))
    print("  Batch size = {}".format(args.eval_batch_size))

    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.bool)
    eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    eval_loss = 0
    step = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_fnt = CrossEntropyLoss()
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc='Evaluation'):
        step += 1
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()
        with torch.no_grad():
            logits = model(input_ids, mask=input_mask) 
            loss = loss_fnt(logits, label_ids)
        eval_loss += loss.mean().item()  # 统计一个batch的损失 一个累加下去
        labels = label_ids.data.cpu().numpy()
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

    # Prepare Optimizer
    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    model = MemoryTransformerXL(
        num_tokens=21129,   # 21128
        dim=1024,
        heads=8,
        depth=8,
        seq_len=1024,
        mem_len=256,            # short term memory (the memory from transformer-xl)
        lmem_len=256,           # long term memory (memory attention network attending to short term memory and hidden activations)
        mem_write_iters=2,      # number of iterations of attention for writing to memory
        memory_layers=[6, 7, 8],  # which layers to use memory, only the later layers are actually needed
        num_mem_kv=128,         # number of memory key/values, from All-attention paper
        )

    # 模型
    if torch.cuda.is_available():
        model.cuda()
         
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
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.bool)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        model.train()
        loss_fnt = CrossEntropyLoss()
        for epoch in range(args.num_train_epochs):
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
            for step, batch in enumerate(train_dataloader):
                start_time = time.time()
                batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, labels_ids = batch
                # input_ids： batch_size, max_len
                # input_mask: batch_size, max_len   注意 这里是bool类型
                logits = model(input_ids, mask=input_mask)   
                loss = loss_fnt(logits, labels_ids)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                print('transformer_xl-model*****epoch:{}, step:{}, loss:{:10f}, time_cost:{:10f}'.format(epoch, step, loss, time.time()-start_time))

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
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

