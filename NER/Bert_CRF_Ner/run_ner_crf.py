# -*- coding: utf-8 -*-
# @Time    : 2020/9/4 14:33
# @Author  : xiaolu
# @FileName: run_ner_crf.py
# @Software: PyCharm
import os
import json
import time
import torch
import torch.nn as nn
from os.path import join
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from data_process import CnerProcessor as Processor
from metrics import SeqEntityScore
from model import BertCrfForNer
from transformers import BertTokenizer
from data_process import convert_examples_to_features
from data_process import collate_fn, get_entities
from config import get_argparse


def json_to_text(file_path,data):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


def load_and_cache_examples(args, tokenizer, data_type='train'):
    # 加载数据
    save_processed_data = './data/{}_processed_data'.format(data_type)
    if os.path.exists(save_processed_data):
        print('加载数据')
        features = torch.load(save_processed_data)
    else:
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.max_seq_length,
                                                cls_token_at_end=False,
                                                pad_on_left=False,
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=0,
                                                sep_token=tokenizer.sep_token,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )

        torch.save(features, save_processed_data)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


def evaluate(args, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label)
    eval_dataset = load_and_cache_examples(args, tokenizer, data_type='dev')

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    print("***** Running evaluation %s *****", prefix)
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}

            outputs = model(**inputs)

            tmp_eval_loss, logits = outputs[:2]

            tags = model.crf.decode(logits, inputs['attention_mask'])

        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        input_lens = inputs['input_lens'].cpu().numpy().tolist()
        tags = tags.squeeze(0).cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(args.id2label[tags[i][j]])
    print("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    print("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    print(info)
    print("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        print("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        print(info)
    return results


if __name__ == "__main__":
    args = get_argparse().parse_args()
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.output_dir):  # 输出文件
        os.mkdir(args.output_dir)

    processor = Processor()
    label_list = processor.get_labels()
    # 将标签进行id映射
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain/vocab.txt')

    # 实例化模型
    model = BertCrfForNer(num_labels)
    model.to(device)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=collate_fn)
        t_total = len(train_dataset) // args.gradient_accumulation_steps * args.epochs

        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(model.bert.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.learning_rate},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_learning_rate},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.crf_learning_rate}
        ]

        # 定义优化器
        warmup_steps = int(t_total * args.warmup_proportion)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        print("***** Running training *****")
        print("  Num examples = %d" % len(train_dataset))
        print("  Num Epochs = %d" % args.epochs)
        print("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        print("  Total optimization steps = %d", t_total)
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        model.train()
        for epoch in range(int(args.epochs)):
            for step, batch in enumerate(train_dataloader):
                start_time = time.time()
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3], 'input_lens': batch[4]}
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                print('epoch:{}, step:{}, loss:{:10f}, time:{:10f}'.format(epoch, step, loss, time.time() - start_time))

                if (global_step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1

            result = evaluate(args, model, tokenizer, prefix="")
            print(result)

            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), join(args.checkpoint_path, "ckpt_epoch_{}.bin".format(epoch)))


