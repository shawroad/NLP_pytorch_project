# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 20:56
# @Author  : xiaolu
# @FileName: train_bert.py
# @Software: PyCharm
import os
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from transformers import BertModel, BertPreTrainedModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
import argparse


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id


class Processor(object):
    def get_train_examples(self, data_dir):
        # 加载训练数据
        return self._create_examples(os.path.join(data_dir, 'train.txt'), 'train')

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'test.txt'), 'test')

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'), 'dev')

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, data_path, set_type):
        examples = []
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                label, text = line.strip().split('\t', 1)
                guid = "{0}-{1}-{2}".format(set_type, label, i)
                examples.append(InputExample(guid=guid, text=text, label=label))
        random.shuffle(examples)
        return examples


def convert_examples_to_features(examples, label_list, max_seq, tokenizer):
    # 文本token转为id序列
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for ex_index, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        tokens = ["[CLS]"] + tokens[:max_seq - 2] + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq - len(input_ids))
        label_id = label_map[example.label]
        features.append(InputFeatures(
            input_ids=input_ids + padding,
            input_mask=input_mask + padding,
            label_id=label_id))
    return features


class BertClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, input_mask, label_ids):
        _, pooled_output = self.bert(input_ids, None, input_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
        return logits


class BertTextCNN(BertPreTrainedModel):
    def __init__(self, config, hidden_size=128, num_labels=2):
        super(BertTextCNN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.conv1 = nn.Conv2d(1, hidden_size, (3, config.hidden_size))
        self.conv2 = nn.Conv2d(1, hidden_size, (4, config.hidden_size))
        self.conv3 = nn.Conv2d(1, hidden_size, (5, config.hidden_size))
        self.classifier = nn.Linear(hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_mask, label_ids):
        sequence_output, _ = self.bert(input_ids, None, input_mask, output_all_encoded_layers=False)
        out = self.dropout(sequence_output).unsqueeze(1)
        c1 = torch.relu(self.conv1(out).squeeze(3))
        p1 = F.max_pool1d(c1, c1.size(2)).squeeze(2)
        c2 = torch.relu(self.conv2(out).squeeze(3))
        p2 = F.max_pool1d(c2, c2.size(2)).squeeze(2)
        c3 = torch.relu(self.conv3(out).squeeze(3))
        p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)
        pool = self.dropout(torch.cat((p1, p2, p3), 1))
        logits = self.classifier(pool)
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
        return logits


def compute_metrics(preds, labels):
    return {'ac': (preds == labels).mean(), 'f1': f1_score(y_true=labels, y_pred=preds)}


def main(args):
    processor = Processor()
    train_examples = processor.get_train_examples(args.data_dir)   # 加载训练集
    label_list = processor.get_labels()    # 加载标签

    # 将token转id
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model, do_lower_case=True)

    train_features = convert_examples_to_features(train_examples, label_list, args.max_seq, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    model = BertClassification.from_pretrained(args.pretrain_model, num_labels=len(label_list))
    # model = BertTextCNN.from_pretrained(args.pretrain_model, num_labels=len(label_list))  # Bert+CNN

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    warmup_steps = t_total * args.warmup_rate
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    model.train()
    global_step = 0
    print('start training...')
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            loss = model(input_ids, input_mask, label_ids)
            # loss.backward()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            print('epoch: {}, step: {}, loss: {:10f}'.format(epoch, step, loss))

            # 梯度裁剪会影响准确率  所以 尽量不用
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            global_step += 1

        print('start evaluating...')
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq, tokenizer)
        eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

        model.eval()
        preds = []
        for i, batch in enumerate(eval_dataloader):
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = model(input_ids, input_mask, None)
                preds.append(logits.detach().cpu().numpy())
        preds = np.argmax(np.vstack(preds), axis=1)
        res = compute_metrics(preds, eval_label_ids.numpy())
        temp = 'epoch:{}, accuracy:{:10f}, f1:{:10f}'.format(epoch, res['ac'], res['f1'])
        print(temp)
        with open('log.txt', 'a+', encoding='utf8') as f:
            temp += '\n'
            f.write(temp)

        output_model_file = os.path.join(args.checkpoint_path, "pytorch_model_epoch{}.bin".format(epoch))
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        print("model save in %s" % output_model_file)
        torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', )

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2, )
    parser.add_argument('--max_seq', type=int, default=128, )

    parser.add_argument('--device', type=str, default='0', help='this is learning of train')
    parser.add_argument('--pretrain_model', type=str, default='./bert_base_pretrain', )
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='this is learning of train')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='this is learning of train')
    parser.add_argument('--warmup_rate', type=int, default=0.05, help='this is learning of train')

    parser.add_argument('--checkpoint_path', type=str, default='./save_model/', help='model save path')
    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_path, exist_ok=True)
    main(args)
