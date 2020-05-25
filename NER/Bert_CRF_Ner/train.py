"""

@file  : train.py

@author: xiaolu

@time  : 2020-05-25

"""
from torch import nn
from pytorch_pretrained_bert import BertAdam
import time
import torch
from data_loader import DataLoader
from model import Model
from config import Config
from rlog import _log_normal, _log_warning, _log_info, _log_error, _log_toomuch, _log_bg_blue, _log_bg_pp, _log_fg_yl, \
    _log_fg_cy, _log_black, rainbow


def evaluate(model, val_data):
    model.eval()
    val_size = val_data['size']
    val_data_iterator = data_loader.data_iterator(val_data, shuffle=True)
    val_iter = val_size // Config.batch_size - 1
    count = 0
    y_predicts, y_labels = [], []
    eval_loss, eval_acc, eval_f1 = 0, 0, 0

    with torch.no_grad():
        for i in range(val_iter):
            batch_data, batch_tags = next(val_data_iterator)
            # print(batch_data.size())   # torch.Size([2, 52])
            # print(batch_tags.size())    # torch.Size([2, 52])
            batch_masks = batch_data.gt(0)
            bert_encode = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
            eval_los = model.loss_fn(bert_encode=bert_encode, tags=batch_tags, output_mask=batch_masks)
            eval_loss += eval_los
            count += 1
            predicts = model.predict(bert_encode, batch_masks)
            y_predicts.append(predicts)

            label_ids = batch_tags.view(1, -1)
            label_ids = label_ids[label_ids != -1]
            y_labels.append(label_ids)

        eval_predicted = torch.cat(y_predicts, dim=0).cpu()
        eval_labeled = torch.cat(y_labels, dim=0).cpu()
        model.class_report(eval_predicted, eval_labeled)

        eval_acc, eval_f1 = model.acc_f1(eval_predicted, eval_labeled)
        print('eval_loss:{:5f}, eval_acc:{:5f}, eval_f1:{:5f}'.format(eval_loss/count, eval_acc, eval_f1))

        return eval_f1


if __name__ == '__main__':
    # Initialize the DataLoader
    data_loader = DataLoader()

    # Load training data and test data
    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('val')

    # Specify the training and validation dataset sizes
    train_size = train_data['size']
    val_size = val_data['size']
    # print(train_size)    # 42000
    # print(val_size)   # 3000

    # Prepare model
    model = Model()
    model.to(Config.device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # 这里我们用bertAdam优化器
    train_iter = train_size // Config.batch_size - 1

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=Config.learning_rate,
                         warmup=0.05,
                         t_total=train_iter * Config.epoch_num)

    best_f1 = 0
    model.train()
    model.to(Config.device)
    for epoch in range(1, Config.epoch_num + 1):
        start_time = time.time()
        # Run one epoch
        print("Epoch {}/{}".format(epoch, Config.epoch_num))
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        for i in range(train_iter):
            start_time = time.time()
            batch_data, batch_tags = next(train_data_iterator)
            # print(batch_data.size())   # torch.Size([2, 52])
            # print(batch_tags.size())    # torch.Size([2, 52])
            batch_masks = batch_data.gt(0)

            bert_encode = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
            train_loss = model.loss_fn(bert_encode=bert_encode, tags=batch_tags, output_mask=batch_masks)
            train_loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=Config.clip_grad)
            # performs updates using calculated gradients
            optimizer.step()

            predicts = model.predict(bert_encode, batch_masks)
            label_ids = batch_tags.view(1, -1)
            label_ids = label_ids[label_ids != -1]
            label_ids = label_ids.cpu()
            train_acc, f1 = model.acc_f1(predicts, label_ids)
            s = "Epoch:{}, step:{}, loss:{:8f}, acc:{:5f}, f1:{:5f}, spend_time:{:6f}".format(
                    epoch, i, train_loss, train_acc, f1, time.time()-start_time)
            rainbow(s)

        if epoch % 1 == 0:
            eval_f1 = evaluate(model, val_data)
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                torch.save(model.state_dict(), './save_model/' + 'best_model.bin')
