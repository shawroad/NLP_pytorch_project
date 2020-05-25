"""

@file  : train.py

@author: xiaolu

@time  : 2020-05-25

"""
from pytorch_pretrained_bert import BertAdam
from torch import nn
import torch
import numpy as np
from data_loader import DataLoader
from model import BertSoftmaxForNer
from config import Config


def evaluate(model, val_data):
    model.eval()
    val_size = val_data['size']
    val_iter = val_size // Config.batch_size - 1
    val_data_iterator = data_loader.data_iterator(val_data, shuffle=True)
    losses = []
    with torch.no_grad():
        for i in range(val_iter):
            batch_data, batch_tags = next(val_data_iterator)
            # print(batch_data.size())   # torch.Size([2, 52])
            # print(batch_tags.size())    # torch.Size([2, 52])
            batch_masks = batch_data.gt(0)
            loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
            losses.append(loss.item())
    return np.mean(losses)


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
    model = BertSoftmaxForNer()
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

    best_val_f1 = 0.0
    patience_counter = 0
    model.train()
    model.to(Config.device)
    best_loss = float('inf')
    for epoch in range(1, Config.epoch_num + 1):
        # Run one epoch
        print("Epoch {}/{}".format(epoch, Config.epoch_num))
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        for i in range(train_iter):
            batch_data, batch_tags = next(train_data_iterator)
            # print(batch_data.size())   # torch.Size([2, 52])
            # print(batch_tags.size())    # torch.Size([2, 52])
            batch_masks = batch_data.gt(0)

            # compute model output and loss
            loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
            print("Epoch:{}, step:{}, loss:{}".format(epoch, i, loss))
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=Config.clip_grad)
            optimizer.step()
        if epoch % 10 == 0:
            # train_metrics = evaluate(model, train_data_iterator, params, mark='Train')
            val_loss = evaluate(model, val_data)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), './save_model/' + 'best_model.bin')
                model.train()
                print('val_loss: {}'.format(val_loss))
