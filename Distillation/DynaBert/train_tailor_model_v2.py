# -*- encoding: utf-8 -*-
'''
@File    :   train_tailor_model_v2.py
@Time    :   2020/10/19 15:16:03
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import os
import torch
import gzip
import pickle
import time
import random
import math
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from rlog import rainbow
from sklearn.metrics import accuracy_score, recall_score, precision_score
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from my_transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from config_tailor import set_args
from model_teacher import Teacher_Model
from torch.nn import  MSELoss


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = ""
        s += "input_id: %s"%((self.input_ids))
        s += ", input_mask: %s"%(self.input_mask)
        s += ", segment_ids: %s"%(self.segment_ids)
        s += ", label_id: %s"%(self.label_id)
        return s


def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return -torch.sum(targets_prob * student_likelihood, dim=-1).mean()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def train(args, train_features, model, tokenizer, eval_features, teacher_model):
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
        ]
        
    warmup_steps = 0.05 * t_total
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    set_seed(args)
    loss_mse = MSELoss()
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.cuda() for t in batch)
            teacher_inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3], 'segment_ids': batch[2]}
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3], 'token_type_ids': batch[2]}

            with torch.no_grad():
                teacher_logits, layer_13_output = teacher_model(**teacher_inputs)

            start_time = time.time()
            # 先对高度进行缩减  再对宽度进行缩减

            # accumulate grads for all sub-networks
            for depth_mult in sorted(args.depth_mult_list, reverse=True):
                model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))
                n_layers = model.config.num_hidden_layers

                depth = round(depth_mult * n_layers)
                kept_layers_index = []
                for i in range(depth):
                    kept_layers_index.append(math.floor(i / depth_mult))
                kept_layers_index.append(n_layers)
                s = ''
                width_idx = 0
                for width_mult in sorted(args.width_mult_list, reverse=True):
                    model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                    
                    loss, student_logit, student_reps, _, _ = model(**inputs)
                    logit_loss = soft_cross_entropy(student_logit, teacher_logits.detach())
                    # loss = args.width_lambda1 * logit_loss + args.width_lambda2 * rep_loss
                    # loss = logit_loss   # 这里只加入蒸馏最终的损失  不加入层损失
                    # for student_rep, teacher_rep in zip(student_reps, list(layer_13_output[i] for i in kept_layers_index)):
                    #     print('------------------------------------')
                    #     print(student_reps)
                    #     print('*************************************')
                    #     print(teacher_rep)
                    #     print('------------------------------------')
                    #     # print(student_reps.size())
                    #     # print(teacher_rep.size())
                    #     exit()
                        
                    #     tmp_loss = loss_mse(student_reps, teacher_rep.detach())
                    #     rep_loss += tmp_loss
                    # loss = logit_loss + rep_loss
                    loss = logit_loss

                    s += 'width={}: {}, '.format(width_mult, loss)
                    if args.n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()

                t = 'tailoring*****epoch:{}, step:{}, depth:{}, {}time:{}'.format(epoch, step, len(kept_layers_index), s, time.time()-start_time)
                rainbow(t)
        exit()
        # clip the accumulated grad from all widths
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        tr_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            # evaluate
            current_best = 0
            if global_step > 0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                acc = []
                for depth_mult in sorted(args.depth_mult_list, reverse=True):
                    model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))
                    for width_mult in sorted(args.width_mult_list, reverse=True):
                        model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                        eval_loss, eval_accuracy = evaluate(args, model, tokenizer, eval_features)
                        acc.append(eval_accuracy)
                    if sum(acc) > current_best:
                        current_best = sum(acc)
                        os.makedirs(args.save_student_model, exist_ok=True)
                        print('Saving model checkpoint to %s'%(args.save_teacher_model))
                        model_to_save = model.modules if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(args.save_student_model)
                        torch.save(args, os.path.join(args.save_student_model, 'student_model.bin'))
                        model_to_save.config.to_json_file(os.path.join(args.save_student_model, 'config.json'))
                        tokenizer.save_vocabulary(args.save_student_model)

def evaluate(args, model, tokenizer, eval_features):
    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss = 0
    step = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_fnt = nn.CrossEntropyLoss()
    model.eval()
    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        step += 1
        batch = tuple(t.cuda() for t in batch)
        label_ids = batch[3].cuda()
        
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3], 'token_type_ids': batch[2]}
        loss, logits, student_rep, _, _ = model(**inputs)
        eval_loss += loss.mean().item()

        labels = label_ids.data.cpu().numpy()
        predic = torch.max(logits.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)
        
    # 损失 召回率 查准率
    eval_loss = eval_loss / step
    eval_accuracy = accuracy_score(labels_all, predict_all)
    return eval_loss, eval_accuracy


def compute_neuron_head_importance(args, model, tokenizer, eval_features):
    model = model.module if hasattr(model, 'module') else model
    base_model = getattr(model, model.base_model_prefix, model)

    # 对头
    n_layers, n_heads = base_model.config.num_hidden_layers, base_model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).cuda()
    head_mask = torch.ones(n_layers, n_heads).cuda()
    head_mask.requires_grad_(requires_grad=True)

    # 对前馈网络
    intermediate_weight = []
    intermediate_bias = []
    output_weight = []
    for name, w in model.named_parameters():
        if 'intermediate' in name:
            if w.dim() > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if 'output' in name and 'attention' not in name:
            if w.dim() > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(torch.zeros(w.shape[0]).cuda())

    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, head_mask=head_mask)
        loss = outputs[0]
        loss.backward()
        head_importance += head_mask.grad.abs().detach()
        for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight, neuron_importance):
            current_importance += ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).abs().detach()
            current_importance += ((w2 * w2.grad).sum(dim=0)).abs().detach()
    return head_importance, neuron_importance


def reorder_neuron_head(model, head_importance, neuron_importance):
    """
    model: bert model
    head_importance: 12*12 matrix for head importance in 12 layers
    neuron_importance: list for neuron importance in 12 layers.
    """
    model = model.module if hasattr(model, 'module') else model
    base_model = getattr(model, model.base_model_prefix, model)

    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = torch.sort(head_importance[layer], descending=True)[-1]    # 对每一层的多个头进行排序
        base_model.encoder.layer[layer].attention.reorder_heads(idx)
        # reorder neurons
        idx = torch.sort(current_importance, descending=True)[-1]
        base_model.encoder.layer[layer].intermediate.reorder_neurons(idx)
        base_model.encoder.layer[layer].output.reorder_neurons(idx)


def main():
    args = set_args()

    # 加载训练集
    with gzip.open(args.train_data_path, 'rb') as f:
        train_features = pickle.load(f)
    
    # 加载验证集
    with gzip.open(args.dev_data_path, 'rb') as f:
        eval_features = pickle.load(f)

    args.width_mult_list = [float(width) for width in args.width_mult_list.split(',')]   # 宽度瘦身的比重  
    args.depth_mult_list = [float(depth) for depth in args.depth_mult_list.split(',')]   # 高度减小的力度
    
    tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)

    teacher_model = Teacher_Model(args)
    teacher_model.load_state_dict(torch.load('./save_teacher_model/best_pytorch_model.bin'))
    
    # 学生模型 直接为预训练的分类模型
    config = BertConfig.from_pretrained(args.model_config, num_labels=2)
    config.output_attentions, config.output_hidden_states, config.output_intermediate = True, True, True
    model = BertForSequenceClassification.from_pretrained(args.pretrain_model, config=config)   # 学生模型

    if torch.cuda.is_available():
        teacher_model.cuda()
        model.cuda()
    # 至此  老师和学生模型构建完毕

    # Set seed
    set_seed(args)

    # 衡量模型中各个头和各个feed forward中的神经元的重要性
    head_importance, neuron_importance = compute_neuron_head_importance(args, model, tokenizer, eval_features)
    reorder_neuron_head(model, head_importance, neuron_importance)
    print('对头和feed forward的重要性衡量完毕...')

    # Training
    train(args, train_features, model, tokenizer, eval_features, teacher_model)



if __name__ == '__main__':
    main()

