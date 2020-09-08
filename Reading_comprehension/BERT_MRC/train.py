"""

@file  : train_RNet.py

@author: xiaolu

@time  : 2020-03-04

"""
import torch
import random
from model import Model
from DataLoader import DatasetIterater, build_dataset
from config import Config
from pytorch_pretrained_bert.optimization import BertAdam


# 随机种子
random.seed(Config.seed)
torch.manual_seed(Config.seed)


train_loss = []
eval_loss = []

def train():
    device = Config.device
    # 准备数据
    train_data, dev_data = build_dataset(Config)
    train_iter = DatasetIterater(train_data, Config)
    dev_iter = DatasetIterater(dev_data, Config)

    model = Model().to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 这里我们用bertAdam优化器

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=Config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * Config.num_epochs)

    model.to(device)
    model.train()

    best_loss = 100000.0
    for epoch in range(Config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, Config.num_epochs))
        for step, batch in enumerate(train_iter):
            input_ids, input_mask, start_positions, end_positions = \
                batch[0], batch[1], batch[2], batch[3]
            input_ids, input_mask, start_positions, end_positions = \
                input_ids.to(device), input_mask.to(device), start_positions.to(device), end_positions.to(device)

            loss, _, _ = model(input_ids, attention_mask=input_mask,
                               start_positions=start_positions, end_positions=end_positions)

            loss.backward()
            optimizer.step()
            print('epoch:{}, step:{}, loss:{}')
            train_loss.append(loss)

            if step % 100 == 0:
                eval_loss = evaluate(model, dev_iter)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    torch.save(model.state_dict(), './save_model/'+'best_model')
                    model.train()


def evaluate(model, dev_iter):
    total, losses = 0.0, []
    device = Config.device

    with torch.no_grad():
        model.eval()
        for batch in dev_iter:
            input_ids, input_mask, start_positions, end_positions = \
                batch[0], batch[1], batch[2], batch[3]
            input_ids, input_mask, start_positions, end_positions = \
                input_ids.to(device), input_mask.to(device), start_positions.to(device), end_positions.to(device)
            loss, _, _ = model(input_ids, input_mask, start_positions, end_positions)

            losses.append(loss.item())

        # 验证集的平均损失
        total_loss = 0.0
        for i in losses:
            total_loss += i

        eval_loss.append(total_loss / len(losses))
        # print("验证集的平均损失:", total_loss / len(losses))
        return total_loss / len(losses)


if __name__ == "__main__":
    train()

    # 保存两种损失
    # 1. 训练损失
    t_loss = '\n'.join([str(loss) for loss in train_loss])
    with open('train_loss.data', 'w') as f:
        f.write(t_loss)

    # 2. 验证集损失
    e_loss = '\n'.join([str(loss) for loss in eval_loss])
    with open('eval_loss.data', 'w') as f:
        f.write(e_loss)


