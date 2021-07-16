"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-07-16
"""
import torch
import numpy as np
from torch import nn
from sklearn import metrics
from config import set_args
# from nezha_model import Model
from nezha_coattention_model import Model
from data_helper import SentencePairDataset
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from utils import focal_loss, FGM
from pdb import set_trace


def train(model, epoch, train_dataloader, test_dataloader, optimizer, scheduler=None):
    print("Training at epoch {}".format(epoch))

    if args.use_fgm:
        print("Using fgm for adversial attack")

    # 一轮总共有多少个batch
    batch_num = len(train_dataloader.dataset) / train_dataloader.batch_size
    model.train()

    if args.use_fgm:
        fgm = FGM(model)

    total_loss = []
    total_gt_a, total_preds_a = [], []
    total_gt_b, total_preds_b = [], []
    for idx, batch in enumerate(train_dataloader):
        source_input_ids, target_input_ids, labels, types = batch   # types的size: batch_size
        if torch.cuda.is_available():
            source_input_ids = source_input_ids.cuda()
            target_input_ids = target_input_ids.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        all_probs = model(source_input_ids, target_input_ids)
        # [torch.Size([batch_size, label_num)]), torch.Size([batch_size, label_num)])]

        num_tasks = len(all_probs)
        all_masks = [(types == task_id).numpy() for task_id in range(num_tasks)]  # 相当于来了个多任务
        # 若batch=4 则all_masks=[array([ True, False, False,  True]), array([False,  True,  True, False])]

        all_output = [all_probs[task_id][all_masks[task_id]] for task_id in range(num_tasks)]
        all_labels = [labels[all_masks[task_id]] for task_id in range(num_tasks)]

        all_loss = None
        for task_id in range(num_tasks):
            if all_masks[task_id].sum() != 0:
                if all_loss is None:
                    all_loss = criterion(all_output[task_id], all_labels[task_id])
                else:
                    all_loss += criterion(all_output[task_id], all_labels[task_id])
        all_loss.backward()

        # code for fgm adversial training
        if args.use_fgm:
            fgm.attack()
            adv_all_probs = model(source_input_ids, target_input_ids)
            adv_all_output = [adv_all_probs[task_id][all_masks[task_id]] for task_id in range(num_tasks)]
            # calculate the loss and BP
            adv_all_loss = None
            for task_id in range(num_tasks):
                if all_masks[task_id].sum() != 0:
                    if adv_all_loss is None:
                        adv_all_loss = criterion(adv_all_output[task_id], all_labels[task_id])
                    else:
                        adv_all_loss += criterion(adv_all_output[task_id], all_labels[task_id])
            adv_all_loss.backward()
            fgm.restore()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        all_gt = [all_labels[task_id].cpu().numpy().tolist() if all_masks[task_id].sum() != 0 else [] for task_id in
                  range(num_tasks)]
        all_preds = [all_output[task_id].argmax(axis=1).cpu().numpy().tolist() if all_masks[task_id].sum() != 0 else []
                     for task_id in range(num_tasks)]

        gt_a, preds_a = [], []
        for task_id in range(0, num_tasks, 2):
            gt_a += all_gt[task_id]
            preds_a += all_preds[task_id]

        gt_b, preds_b = [], []
        for task_id in range(1, num_tasks, 2):
            gt_b += all_gt[task_id]
            preds_b += all_preds[task_id]

        total_preds_a += preds_a
        total_gt_a += gt_a
        total_preds_b += preds_b
        total_gt_b += gt_b
        total_loss.append(all_loss.item())
        # print('a', preds_a, gt_a)
        # print('b', preds_b, gt_b)

        acc_a = metrics.accuracy_score(gt_a, preds_a) if len(gt_a) != 0 else 0
        f1_a = metrics.f1_score(gt_a, preds_a, zero_division=0)
        acc_b = metrics.accuracy_score(gt_b, preds_b) if len(gt_b) != 0 else 0
        f1_b = metrics.f1_score(gt_b, preds_b, zero_division=0)

        # learning rate for bert is the second (the last) parameter group
        writer.add_scalar('train/learning_rate', optimizer.param_groups[-1]['lr'], global_step=epoch * batch_num + idx)
        writer.add_scalar('train/loss', all_loss.item(), global_step=epoch * batch_num + idx)
        writer.add_scalar('train/acc_a', acc_a, global_step=epoch * batch_num + idx)
        writer.add_scalar('train/acc_b', acc_b, global_step=epoch * batch_num + idx)
        writer.add_scalar('train/f1_a', f1_a, global_step=epoch * batch_num + idx)
        writer.add_scalar('train/f1_b', f1_b, global_step=epoch * batch_num + idx)

        print('epoch:{}, step:{}, loss:{}, task_a_acc:{}, task_a_f1:{}, task_b_acc:{}, task_b_f1:{}'.format(
            epoch, idx, all_loss.item(), acc_a, f1_a, acc_b, f1_b
        ))

        global best_dev_loss, best_dev_f1
        # 多少步验证依次 一轮进行两次验证，中间一次和最后一次
        if (idx + 1) % (batch_num // 2) == 0:
            dev_loss, dev_acc_a, dev_acc_b, dev_f1_a, dev_f1_b = eval(model, test_dataloader)
            dev_f1 = (dev_f1_a + dev_f1_b) / 2
            writer.add_scalar('eval/loss', dev_loss, global_step=epoch * batch_num + idx)
            writer.add_scalar('eval/acc_a', dev_acc_a, global_step=epoch * batch_num + idx)
            writer.add_scalar('eval/acc_b', dev_acc_b, global_step=epoch * batch_num + idx)
            writer.add_scalar('eval/f1_a', dev_f1_a, global_step=epoch * batch_num + idx)
            writer.add_scalar('eval/f1_b', dev_f1_b, global_step=epoch * batch_num + idx)

            if dev_loss < best_dev_loss or dev_f1 > best_dev_f1:
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    torch.save(model.state_dict(),
                               args.save_dir + '_epoch_{}_{}_'.format(epoch, args.task_type) + 'loss')
                    print("----------BETTER LOSS, MODEL SAVED-----------")
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    torch.save(model.state_dict(),
                               args.save_dir + '_epoch_{}_{}_'.format(epoch, args.task_type) + 'f1')
                    print("----------BETTER F1, MODEL SAVED-----------")

    loss = np.array(total_loss).mean()

    f1_a = metrics.f1_score(total_gt_a, total_preds_a, zero_division=0)
    f1_b = metrics.f1_score(total_gt_b, total_preds_b, zero_division=0)
    f1 = (f1_a + f1_b) / 2
    print("Average f1 on training set: {:.6f}, f1_a: {:.6f}, f1_b: {:.6f}".format(f1, f1_a, f1_b))

    return loss, f1, f1_a, f1_b


def eval(model, test_dataloader):
    print("Evaluating")
    model.eval()

    total_loss = []
    total_gt_a, total_preds_a = [], []
    total_gt_b, total_preds_b = [], []

    for idx, batch in enumerate(test_dataloader):
        source_input_ids, target_input_ids, labels, types = batch
        if torch.cuda.is_available():
            source_input_ids = source_input_ids.cuda()
            target_input_ids = target_input_ids.cuda()
            labels = labels.view(-1).cuda()

        with torch.no_grad():
            all_probs = model(source_input_ids, target_input_ids)
            num_tasks = len(all_probs)

            all_masks = [(types == task_id).numpy() for task_id in range(num_tasks)]
            all_output = [all_probs[task_id][all_masks[task_id]] for task_id in range(num_tasks)]
            all_labels = [labels[all_masks[task_id]] for task_id in range(num_tasks)]

            all_loss = None
            for task_id in range(num_tasks):
                if all_masks[task_id].sum() != 0:
                    if all_loss is None:
                        all_loss = criterion(all_output[task_id], all_labels[task_id])
                    else:
                        all_loss += criterion(all_output[task_id], all_labels[task_id])

            all_gt = [all_labels[task_id].cpu().numpy().tolist() if all_masks[task_id].sum() != 0 else [] for task_id in
                      range(num_tasks)]
            all_preds = [
                all_output[task_id].argmax(axis=1).cpu().numpy().tolist() if all_masks[task_id].sum() != 0 else [] for
                task_id in range(num_tasks)]

            gt_a, preds_a = [], []
            for task_id in range(0, num_tasks, 2):
                gt_a += all_gt[task_id]
                preds_a += all_preds[task_id]

            gt_b, preds_b = [], []
            for task_id in range(1, num_tasks, 2):
                gt_b += all_gt[task_id]
                preds_b += all_preds[task_id]

            total_preds_a += preds_a
            total_gt_a += gt_a
            total_preds_b += preds_b
            total_gt_b += gt_b
            total_loss.append(all_loss.item())

    loss = np.array(total_loss).mean()
    acc_a = metrics.accuracy_score(total_gt_a, total_preds_a) if len(total_gt_a) != 0 else 0
    f1_a = metrics.f1_score(total_gt_a, total_preds_a, zero_division=0)
    if f1_a == 0:
        print("F1_a = 0, checking precision, recall, fscore and support...")
        print(metrics.precision_recall_fscore_support(total_gt_a, total_preds_a, zero_division=0))

    acc_b = metrics.accuracy_score(total_gt_b, total_preds_b) if len(total_gt_b) != 0 else 0
    f1_b = metrics.f1_score(total_gt_b, total_preds_b, zero_division=0)

    if f1_b == 0:
        print("F1_b = 0, checking precision, recall, fscore and support...")
        print(metrics.precision_recall_fscore_support(total_gt_b, total_preds_b, zero_division=0))

    print("Loss on dev set: ", loss)
    print("F1 on dev set: {:.6f}, f1_a: {:.6f}, f1_b: {:.6f}".format((f1_a + f1_b) / 2, f1_a, f1_b))

    return loss, acc_a, acc_b, f1_a, f1_b


if __name__ == '__main__':
    args = set_args()

    # task_a = ['短短匹配A类', '短长匹配A类', '长长匹配A类']
    # task_b = ['短短匹配B类', '短长匹配B类', '长长匹配B类']
    task_a = ['短短匹配A类']
    task_b = ['短短匹配B类']

    # print_every = config.print_every
    # eval_every = config.eval_every

    train_data_dir, dev_data_dir = [], []
    if 'a' in args.task_type:
        for task in task_a:
            train_data_dir.append(args.data_dir + task + '/train.txt')
            dev_data_dir.append(args.data_dir + task + '/valid.txt')

    if 'b' in args.task_type:
        for task in task_b:
            train_data_dir.append(args.data_dir + task + '/train.txt')
            dev_data_dir.append(args.data_dir + task + '/valid.txt')

    # print(train_data_dir)   # 几种训练集的路径
    # print(dev_data_dir)   # 几种验证集的路径

    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    # 加载数据集
    train_dataset = SentencePairDataset(train_data_dir, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    dev_dataset = SentencePairDataset(dev_data_dir, is_train=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # # 如果bert+其他模型，可以将其他模型的学习率设置的不一样 如下:
    # optimizer = AdamW([
    #     {"params": model.all_classifier.parameters(), "lr": classifer_lr},
    #     {"params": model.bert.parameters()}],
    #     lr=args.learning_rate,
    #     weight_decay=args.weight_decay
    # )

    # 总的训练步数
    total_steps = len(train_dataloader) * args.epochs

    if args.use_scheduler:
        # 线性学习率衰减
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=total_steps,
            num_warmup_steps=0.05 * total_steps,  # 预热多少步
        )
    else:
        scheduler = None

    # 损失函数
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    # criterion = focal_loss()    # focal loss损失

    print("Training on Task {}...".format(args.task_type))
    writer = SummaryWriter('runs/{}'.format(args.task_type))

    best_dev_loss = 999
    best_dev_f1 = 0

    for epoch in range(args.epochs):
        train_loss, train_f1, train_f1_a, train_f1_b = train(
            model, epoch, train_dataloader, dev_dataloader, optimizer, scheduler=scheduler
        )
