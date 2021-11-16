"""
@file   : run_sentencebert.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-02
"""
import os
import gzip
import pickle
import torch
import time
import numpy as np
from tqdm import tqdm
from torch import nn
from config import set_args
from model import SentenceBert
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from utils import compute_corrcoef, l2_normalize
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup


class Features:
    def __init__(self, s1_input_ids=None, s2_input_ids=None, label=None):
        self.s1_input_ids = s1_input_ids
        self.s2_input_ids = s2_input_ids
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "s1_input_ids: %s" % (self.s1_input_ids)
        s += ", s2_input_ids: %s" % (self.s2_input_ids)
        s += ", label: %d" % (self.label)
        return s


def evaluate(model):
    model.eval()
    # 语料向量化
    all_a_vecs, all_b_vecs = [], []
    all_labels = []
    for step, batch in tqdm(enumerate(dev_dataloader)):
        s1_input_ids, s2_input_ids, label = batch
        if torch.cuda.is_available():
            s1_input_ids = s1_input_ids.cuda()
            s2_input_ids = s2_input_ids.cuda()
            label = label.cuda()
        with torch.no_grad():
            s1_embeddings, s2_embeddings = model(s1_input_ids=s1_input_ids, s2_input_ids=s2_input_ids)
            s1_embeddings = s1_embeddings.cpu().numpy()
            s2_embeddings = s2_embeddings.cpu().numpy()
            label = label.cpu().numpy()
           
            all_a_vecs.extend(s1_embeddings)
            all_b_vecs.extend(s2_embeddings)
            all_labels.extend(label)

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)

    a_vecs = l2_normalize(all_a_vecs)
    b_vecs = l2_normalize(all_b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(all_labels, sims)
    return corrcoef


def calc_loss(s1_vec, s2_vec, true_label):
    loss_fct = nn.MSELoss()
    output = torch.cosine_similarity(s1_vec, s2_vec)
    loss = loss_fct(output, true_label)
    return loss


if __name__ == '__main__':
    args = set_args()
    args.output_dir = 'checkpoint_base_reg_pos_7k_neg_3k'
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载数据集
    with gzip.open(args.train_data_path, 'rb') as f:
        train_features = pickle.load(f)


    # 开始训练
    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_features)))
    print("  Batch size = {}".format(args.train_batch_size))
    all_s1_input_ids = torch.tensor([f.s1_input_ids for f in train_features], dtype=torch.long)
    all_s2_input_ids = torch.tensor([f.s2_input_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in train_features], dtype=torch.float32)

    train_data = TensorDataset(all_s1_input_ids, all_s2_input_ids, all_label_ids)

    best_corrcoef = None
    global_step = 0
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sign = 0
    for train_index, val_index in kfold.split(all_s1_input_ids, all_label_ids):
        sign += 1
        # 总共训练的步数
        num_train_steps = int(
             len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # 模型
        model = SentenceBert()

        # 指定多gpu运行
        if torch.cuda.is_available():
            model.cuda()

        tokenizer = AutoTokenizer.from_pretrained('./pretrain_weight')

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


        train_data = TensorDataset(all_s1_input_ids[train_index], all_s2_input_ids[train_index], all_label_ids[train_index])
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
        
        dev_data = TensorDataset(all_s1_input_ids[val_index], all_s2_input_ids[val_index], all_label_ids[val_index])
        dev_dataloader = DataLoader(dev_data, batch_size=args.train_batch_size)
        
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                start_time = time.time()
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                s1_input_ids, s2_input_ids, label = batch
                s1_vec, s2_vec = model(s1_input_ids, s2_input_ids)

                loss = calc_loss(s1_vec, s2_vec, label)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                print('KFold:{}, Epoch:{}, Step:{}, Loss:{:10f}, Time:{:10f}'.format(sign, epoch, step, loss, time.time() - start_time))
                loss.backward()

                # nn.utils.clip_grad_norm(model.parameters(), max_norm=20, norm_type=2)   # 是否进行梯度裁剪

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                # corrcoef = evaluate(model)

            # 一轮跑完 进行eval
            corrcoef = evaluate(model)
            ss = 'KFlod:{}, epoch:{}, corrcoef:{}'.format(sign, epoch, corrcoef)
            with open(args.output_dir + '/logs.txt', 'a+', encoding='utf8') as f:
                ss += '\n'
                f.write(ss)

            model.train()
            if best_corrcoef is None or best_corrcoef < corrcoef:
                best_corrcoef = corrcoef
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "best_pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)

            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "kfold_{}_epoch{}_ckpt.bin".format(sign, epoch))
            torch.save(model_to_save.state_dict(), output_model_file)


