"""

@file  : train.py

@author: xiaolu

@time  : 2020-05-28

"""
import argparse
import torch
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from model.model_fastbert import BertConfig, FastBertModel
from utils import load_json_config, load_saved_model, init_bert_adam_optimizer, save_model
from data_utils.dataset_preparing import PrepareDataset, TextCollate


seed = 9999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = torch.device('cuda: 1' if torch.cuda.is_available() else 'cpu')
debug_break = False


def eval_model(train_stage, model, dataset, batch_size=1, num_workers=1):
    global global_step
    global debug_break
    model.eval()
    dataloader = data.DataLoader(dataset=dataset, collate_fn=TextCollate(dataset),
                                 batch_size=batch_size, num_workers=num_workers, shuffle=False)
    total_loss = 0.0
    correct_sum = 0
    proc_sum = 0
    num_batch = dataloader.__len__()
    print('Evaluating Model...')
    for step, batch in enumerate(dataloader):
        tokens = batch['tokens'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        attn_masks = batch['attn_masks'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            loss, logits = model(tokens, token_type_ids=segment_ids, attention_mask=attn_masks, labels=labels,
                                 training_stage=train_stage, inference=False)
        loss = loss.mean()
        loss_val = loss.item()
        total_loss += loss_val
        if debug_break and step > 50:
            break
        if train_stage == 0:
            _, top_index = logits.topk(1)
            correct_sum += (top_index.view(-1) == labels).sum().item()
            proc_sum += labels.shape[0]
    print('eval total avg loss: {}'.format(total_loss / num_batch))
    if train_stage == 0:
        print("Correct Prediction: " + str(correct_sum))
        print("Accuracy Rate: " + format(correct_sum / proc_sum, "0.4f"))


def train_epoch(train_stage, model, optimizer, dataloader, gradient_accumulation_steps, epoch, dump_info=False):
    global global_step
    global debug_break
    model.train()
    dataloader.dataset.is_training = True

    total_loss = 0.0
    correct_sum = 0
    proc_sum = 0
    num_batch = dataloader.__len__()
    for step, batch in enumerate(dataloader):
        tokens = batch['tokens'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        attn_masks = batch['attn_masks'].to(device)
        labels = batch['labels'].to(device)
        # print(tokens.size())   # torch.Size([2, 47])
        # print(labels.size())   # torch.Size([2])
        loss, logits = model(tokens, token_type_ids=segment_ids, attention_mask=attn_masks, labels=labels,
                             training_stage=train_stage, inference=False)

        if train_stage == 0 and dump_info:
            probs = F.softmax(logits, dim=-1)

        loss = loss.mean()

        if gradient_accumulation_steps > 1:
            loss /= gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()

        loss_val = loss.item()
        total_loss += loss_val

        print('epoch:{}, step:{}, loss:{}'.format(epoch, step, loss_val))

        if train_stage == 0:
            _, top_index = logits.topk(1)
            correct_sum += (top_index.view(-1) == labels).sum().item()
            proc_sum += labels.shape[0]
        if debug_break and step > 50:
            break

    print('train total avg loss: {}'.format(total_loss / num_batch))

    if train_stage == 0:
        print('correct prediction:', correct_sum)
        print('Accuracy rate:', correct_sum / proc_sum)
    return total_loss / num_batch


def train_model(train_stage, save_model_path, model, optimizer, epochs,train_dataset, eval_dataset,
                batch_size=1, gradient_accumulation_steps=1, num_workers=1):
    print('Start Training'.center(60, '*'))
    training_dataloader = data.DataLoader(dataset=train_dataset, collate_fn=TextCollate(train_dataset),
                                          batch_size=batch_size, num_workers=num_workers, shuffle=True)
    for epoch in range(1, epochs + 1):
        print('train epoch: ' + str(epoch))
        avg_loss = train_epoch(train_stage, model, optimizer, training_dataloader, gradient_accumulation_steps, epoch)
        print('average_loss:{}'.format(avg_loss))

        eval_model(train_stage, model, eval_dataset, batch_size=batch_size, num_workers=num_workers)
        save_model(save_model_path, model, epoch)


def main(args):
    # 1. 加载预定义的一些配置文件
    config = load_json_config(args.model_config_file)
    bert_config = BertConfig.from_json_file(config.get('bert_config_path'))  # bert模型的配置文件

    # 2. 预训练模型的加载
    if args.run_mode == 'train':
        # 第一步的训练训练的是teacher cls
        if args.train_stage == 0:
            model = FastBertModel.load_pretrained_bert_model(bert_config, config,
                                                             pretrained_model_path=config.get('bert_pretrained_model_path'))
            save_model_path_for_train = args.save_model_path
        # 第二步是去蒸馏student cls
        elif args.train_stage == 1:
            model = FastBertModel(bert_config, config)
            load_saved_model(model, args.save_model_path)
            save_model_path_for_train = args.save_model_path_distill
            for name, p in model.named_parameters():
                if 'branch_classifier' not in name:
                    p.requires_grad = False
            print('Teacher Classifier Freezed, Student Classifier will Distilling')
        else:
            print('error, please choose 0 or 1')

    elif args.run_mode == 'eval':
        model = FastBertModel(bert_config, config)
        load_saved_model(model, args.save_model_path)

    else:
        print('Operation mode not legal')

    print("initialize model Done".center(60, '*'))
    model.to(device)

    # 3. 数据集的初始化
    if args.train_data:
        train_dataset = PrepareDataset(vocab_file=config.get('vocab_file'),
                                       max_seq_len=config.get('max_seq_len'),
                                       num_class=config.get('num_class'),
                                       data_file=args.train_data)
        print('load training dataset done. total training num: {}'.format(train_dataset.__len__()))

    if args.eval_data:
        eval_dataset = PrepareDataset(vocab_file=config.get('vocab_file'),
                                      max_seq_len=config.get('max_seq_len'),
                                      num_class=config.get('num_class'),
                                      data_file=args.eval_data)
        print('load eval dataset done. total eval num: {}'.format(eval_dataset.__len__()))

    # 4.开始训练
    if args.run_mode == 'train':
        optimizer = init_bert_adam_optimizer(model, train_dataset.__len__(), args.epochs, args.batch_size,
                                             config.get('gradient_accumulation_steps'), config.get('init_lr'),
                                             config.get('warmup_proportion'))

        train_model(args.train_stage, save_model_path_for_train, model, optimizer,
                    args.epochs, train_dataset, eval_dataset, batch_size=args.batch_size,
                    gradient_accumulation_steps=config.get('gradient_accumulation_steps'),
                    num_workers=args.data_load_num_workers)

    elif args.run_mode == 'eval':
        eval_model(args.train_stage, model, eval_dataset, batch_size=args.batch_size, num_workers=args.data_load_num_workers)
    else:
        print('参数错误')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text classification training script arguments.")
    parser.add_argument("--model_config_file", default='config/fastbert_cls.json', dest="model_config_file", action="store",
                        help="The path of configuration json file.")

    parser.add_argument("--run_mode", default='train', dest="run_mode", action="store",
                        help="Running mode: train or eval")

    # step0 is to train for teacher cls  step1 is to train for student cls
    parser.add_argument("--train_stage", default=0, dest="train_stage", action="store", type=int,
                        help="Running train stage, 0 or 1.")

    parser.add_argument("--save_model_path", default='save_model/fastbert_test', dest="save_model_path", action="store",
                        help="The path of trained checkpoint model.")

    parser.add_argument("--save_model_path_distill", dest="save_model_path_distill", action="store",
                        help="The path of trained checkpoint model.")

    parser.add_argument("--train_data", default='data/ChnSentiCorp/train.tsv', dest="train_data", action="store", help="")

    parser.add_argument("--eval_data", default='data/ChnSentiCorp/train.tsv', dest="eval_data", action="store", help="")

    parser.add_argument("--inference_speed", dest="inference_speed", action="store",
                        type=float, default=1.0, help="")

    # -1 for NO GPU
    parser.add_argument("--gpu_ids", dest="gpu_ids", action="store", default="0",
                        help="Device ids of used gpus, split by ',' , IF -1 then no gpu")

    parser.add_argument("--epochs", default=8, dest="epochs", action="store", type=int,  help="")

    parser.add_argument("--batch_size", default=32, dest="batch_size", action="store", type=int, help="")

    parser.add_argument("--data_load_num_workers", default=2, dest="data_load_num_workers", action="store", type=int,
                        help="")

    parser.add_argument("--debug_break", default=0, dest="debug_break", action="store", type=int,
                        help="Running debug_break, 0 or 1.")

    parsed_args = parser.parse_args()
    # debug_break = (parsed_args.debug_break == 1)
    main(parsed_args)
