# -*- coding: utf-8 -*-
"""
@Time ： 2020/10/30 9:54
@Auth ： xiaolu
@File ：train.py
@IDE ：PyCharm
@Email：luxiaonlp@163.com
"""
import os
import gzip
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from data_process import get_intent_labels, get_slot_labels
from utils import compute_metrics
from model import JointBert
from config import set_args


class InputExample:
    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "guid: %s" % (str(self.guid))
        s += ", words: %s" % (self.words)
        s += ", intent_label: %s" % (self.intent_label)
        s += ", slot_labels: %s" % (self.slot_labels)
        return s


class InputFeatures:
    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "input_ids: %s" % (str(self.input_ids))
        s += ", attention_mask: %s" % (self.attention_mask)
        s += ", token_type_ids: %s" % (self.token_type_ids)
        s += ", intent_label_id: %s" % (self.intent_label_id)
        s += ", slot_labels_ids: %s" % (self.slot_labels_ids)
        return s


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def evaluate(model, eval_features):
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in eval_features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in eval_features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in eval_features], dtype=torch.long)
    dev_dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)
    eval_sampler = SequentialSampler(dev_dataset)
    eval_dataloader = DataLoader(dev_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("***** Running evaluation on dataset *****")
    print("  Num examples = %d", len(dev_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    intent_preds = None
    slot_preds = None
    out_intent_label_ids = None
    out_slot_labels_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'intent_label_ids': batch[3],
                      'slot_labels_ids': batch[4]}
            tmp_eval_loss, intent_logits, slot_logits = model(**inputs)
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        # Intent prediction
        if intent_preds is None:
            intent_preds = intent_logits.detach().cpu().numpy()
            out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
        else:
            intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
            out_intent_label_ids = np.append(
                out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

        # Slot prediction
        if slot_preds is None:
            if args.use_crf:
                # decode() in `torchcrf` returns list with best index directly
                slot_preds = np.array(model.crf.decode(slot_logits))
            else:
                slot_preds = slot_logits.detach().cpu().numpy()

            out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
        else:
            if args.use_crf:
                slot_preds = np.append(slot_preds, np.array(model.crf.decode(slot_logits)), axis=0)
            else:
                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

            out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(),
                                            axis=0)

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }

    intent_preds = np.argmax(intent_preds, axis=1)

    if not args.use_crf:
        slot_preds = np.argmax(slot_preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
    slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

    for i in range(out_slot_labels_ids.shape[0]):
        for j in range(out_slot_labels_ids.shape[1]):
            if out_slot_labels_ids[i, j] != args.ignore_index:
                out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
    results.update(total_result)

    print("***** Eval results *****")
    for key in sorted(results.keys()):
        print("  %s = %s", key, str(results[key]))
    return results['loss']


if __name__ == '__main__':
    args = set_args()
    set_seed(args)

    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    # 加载词表
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)

    # 加载数据
    print('加载训练集...')
    with gzip.open('./processed_data/train_features.pkl.gz', 'rb') as f:
        train_features = pickle.load(f)
    print('加载测试集...')
    with gzip.open('./processed_data/dev_features.pkl.gz', 'rb') as f:
        eval_features = pickle.load(f)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in train_features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in train_features], dtype=torch.long)

    train_dataset = TensorDataset(all_input_ids, all_attention_mask,
                                  all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)
    # 开始训练模型
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)

    config = BertConfig.from_pretrained(args.model_dir)
    model = JointBert.from_pretrained(args.model_dir, config=config, args=args,
                                      intent_label_lst=intent_label_lst, slot_label_lst=slot_label_lst)
    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # Train!
    global_step = 0
    best_loss = None
    for epoch in range(args.num_train_epochs):
        print("***** Running training *****")
        print("  Num examples = %d" % len(train_dataset))
        print("  Num Epochs = %d" % args.num_train_epochs)
        print("  Total train batch size = %d" % args.train_batch_size)
        print("  Gradient Accumulation steps = %d" % args.gradient_accumulation_steps)
        print("  Total optimization steps = %d" % t_total)

        model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'intent_label_ids': batch[3],
                      'slot_labels_ids': batch[4]}
            outputs = model(**inputs)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            print('epoch: {}, step: {}, global_step: {}, loss: {}'.format(epoch, step, global_step, loss))
            test_loss = evaluate(model, eval_features)

        # 保存模型
        test_loss = evaluate(model, eval_features)
        if best_loss is None or best_loss > test_loss:
            best_loss = test_loss
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            os.makedirs(args.save_model, exist_ok=True)
            output_model_file = os.path.join(args.save_model, "best_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.save_model, "epoch{}_ckpt.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)
