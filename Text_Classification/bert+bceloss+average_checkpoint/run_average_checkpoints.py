# -*- encoding: utf-8 -*-
'''
@File    :   run_average_checkpoints.py
@Time    :   2020/10/26 09:13:21
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import os
import torch
from os import listdir
import argparse
import gzip
import pickle
from tqdm import tqdm
import numpy as np
from model import Model
from pdb import set_trace
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    

class CLSExample:
    def __init__(self, doc_id=None, context=None, label=None):
        self.doc_id = doc_id
        self.context = context
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "doc_id: %s" % (str(self.doc_id))
        s += ", context: %s" % (self.context)
        s += ", label: %d" % (self.label)
        return s


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
    
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "input_ids: %s" % (str(self.input_ids))
        s += ", input_mask: %s" % (self.input_mask)
        s += ", segment_ids: %s" % (self.segment_ids)
        s += ", label: %d" % (self.label)
        return s

def evaluate(eval_dataloader, args, model, device):
    loss_fct = BCEWithLogitsLoss()
    model.eval()
    eval_loss = 0
    step = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for input_ids, input_mask, segment_ids, label in eval_dataloader:
        step += 1
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label = label.to(device)

        with torch.no_grad():
            logits = model(input_ids, input_mask, segment_ids, labels=label)
            label = label.view(-1, 1)
            loss = loss_fct(logits, label)
        eval_loss += loss.mean().item()  # 统计一个batch的损失 一个累加下去

        labels = label.data.cpu().numpy()
        
        logits = logits.data.cpu()
        logits = F.sigmoid(logits)
        one = torch.ones(1)
        zero = torch.zeros(1)
        predic = torch.where(logits < 0.5, zero, one).numpy()

        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

    # 损失 召回率 查准率
    eval_loss = eval_loss / step
    eval_accuracy = accuracy_score(labels_all, predict_all)
    return eval_accuracy


def average_checkpoints(checkpoints):
    average_models = {}
    for cp in checkpoints:
        state_dicts = torch.load(cp, map_location='cpu')
        for item in state_dicts:
            average_models[item] = average_models.get(item, torch.tensor(0)) + state_dicts[item]

    for item in average_models:
        average_models[item] /= len(checkpoints)
    return average_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_model", default='./wobert_pretrain/pytorch_model.bin', type=str, help='pretrain_model')
    parser.add_argument("--model_config", default='./wobert_pretrain/bert_config.json', type=str, help='pretrain_model_config')
    parser.add_argument("--vocab_file", default='./wobert_pretrain/vocab.txt')
    parser.add_argument("--train_data_path", default='./data/train_features.pkl.gz', type=str, help='data with training')
    parser.add_argument("--dev_data_path", default='./data/dev_features.pkl.gz', type=str, help='data with dev')
    parser.add_argument('--train_batch_size', type=int, default=256, help="random seed for initialization")
    parser.add_argument('--eval_batch_size', type=int, default=256, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=20, help="random seed for initialization")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="random seed for initialization")
    parser.add_argument('--n_gpu', type=int, default=1, help="random seed for initialization")
    parser.add_argument('--label_num', type=int, default=2, help="label is nums")
    parser.add_argument('--seed', type=int, default=43, help="random seed for initialization")
    # 保存老师模型
    parser.add_argument("--save_model", default='./save_model', type=str)
    parser.add_argument("--output_dir", default='./retrain_model', type=str)
    args = parser.parse_args()

    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    # 加载验证集
    with gzip.open(args.dev_data_path, 'rb') as f:
        eval_features = pickle.load(f)
    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.float32)

    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)


    # 对训练好的模型按编号进行排序
    checkpoint_path = './save_model'
    model_path = []
    for model_name in listdir(checkpoint_path):
        if 'best' in model_name:
            continue
        else:
            model_path.append(os.path.join(checkpoint_path, model_name))
    # 模型加载完毕  然后加载模型  进行推理
    model = Model()
    temp = []
    for path in model_path:
        model.load_state_dict(torch.load(path))
        model = model.to(device)
        eval_accuracy = evaluate(eval_dataloader, args, model, device)
        temp.append([path, eval_accuracy])
    temp.sort(key=lambda x:x[1], reverse=True)

    for model_p, score in temp:
        print('模型: {},  验证集得分: {}'.format(model_p, score))

    best_checkpoints = []
    best_dev_acc = None
    for i, (model_p, score) in enumerate(temp):
        best_checkpoints.append(model_p)
        average_state_dict = average_checkpoints(best_checkpoints)   # 参数平均
        model_to_save = model.modules if hasattr(model, 'module') else model
        model_to_save.load_state_dict(average_state_dict)
        eval_accuracy = evaluate(eval_dataloader, args, model, device)
        print('权重平均{}次以后, 验证集得分: {}'.format(i, eval_accuracy))
    
        torch.save(model_to_save.state_dict(), os.path.join(args.save_model, "average_{}.bin".format(i)))

        if best_dev_acc is None or eval_accuracy > best_dev_acc:
            best_dev_acc = eval_accuracy
            output_model_file = os.path.join(checkpoint_path, "best_average_pytorch_model.bin")
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            print("model save in %s" % output_model_file)
            torch.save(model_to_save.state_dict(), output_model_file)
        
if __name__ == "__main__":
    main()
