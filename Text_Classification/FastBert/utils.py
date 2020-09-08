"""

@file  : utils.py

@author: xiaolu

@time  : 2020-05-28

"""
import json
import os
import collections
import torch
from model.optimization import BERTAdam


def load_json_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_saved_model(model, saved_model_path, model_file=None):
    if model_file is None:
        files = os.listdir(saved_model_path)
        model_file = sorted(files)[-1]
    model_file = os.path.join(saved_model_path, model_file)
    # model_weight = torch.load(model_file, map_location="cpu")
    model_weight = torch.load(model_file)
    new_state_dict = collections.OrderedDict()
    for k, v in model_weight.items():
        if k.startswith("module"):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print('loaded saved model file:', model_file)


def init_bert_adam_optimizer(model, training_data_len, epoch, batch_size,
                             gradient_accumulation_steps, init_lr, warmup_proportion):
    no_decay = ["bias", "gamma", "beta"]
    optimizer_parameters = [
        {"params": [p for name, p in model.named_parameters() if name not in no_decay], "weight_decay_rate": 0.01},
        {"params": [p for name, p in model.named_parameters() if name in no_decay], "weight_decay_rate": 0.0}
    ]
    num_train_steps = int(training_data_len / batch_size / gradient_accumulation_steps * epoch)
    optimizer = BERTAdam(optimizer_parameters,
                         lr=init_lr,
                         warmup=warmup_proportion,
                         t_total=num_train_steps)
    return optimizer


def save_model(path, model, epoch):
    if not os.path.exists(path):
        os.mkdir(path)
    model_weight = model.state_dict()
    new_state_dict = collections.OrderedDict()
    for k, v in model_weight.items():
        if k.startswith("module"):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model_name = "Epoch_" + str(epoch) + ".bin"
    model_file = os.path.join(path, model_name)
    torch.save(new_state_dict, model_file)
    print('dumped model file to:', model_file)


def eval_pr(labels, preds):
    TP, TN, FP, FN = 0, 0, 0, 0
    for label, pred in zip(labels, preds):
        if label == 1 and pred == 1:
            TP += 1
        elif label == 0 and pred == 0:
            TN += 1
        elif label == 1 and pred == 0:
            FP += 1
        elif label == 0 and pred == 1:
            FN += 1
    print('TP', TP)
    print('TN', TN)
    print('FP', FP)
    print('FN', FN)
    precise = TP/(TP+FN)
    recall = TP/(TP+FP)
    return precise, recall
