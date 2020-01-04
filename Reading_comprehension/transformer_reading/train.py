"""

@file  : train.py

@author: xiaolu

@time  : 2020-01-03

"""
import math
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from config import Config, logger

from data_gen import TranslateDataset, pad_collate
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer

from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient


def train_net(args):
    # 为了保证程序执行结果一致, 给随机化设定种子
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint

    start_epoch = 0
    writer = SummaryWriter()

    if checkpoint is None:
        # model
        encoder = Encoder(Config.vocab_size, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)

        decoder = Decoder(Config.sos_id, Config.eos_id, Config.vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)

        model = Transformer(encoder, decoder)

        # optimizer
        optimizer = TransformerOptimizer(
            torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09))

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to GPU, if available
    model = model.to(Config.device)

    # Custom dataloaders  数据的加载　注意这里指定了一个参数collate_fn代表的数据需要padding
    train_dataset = TranslateDataset()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               shuffle=True, num_workers=args.num_workers)

    # Epochs
    Loss_list = []
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger,
                           writer=writer)

        l = str(train_loss)
        Loss_list.append(l)

        l_temp = l + '\n'
        with open('loss_epoch.txt', 'a+') as f:
            f.write(l_temp)

        writer.add_scalar('epoch/train_loss', train_loss, epoch)
        writer.add_scalar('epoch/learning_rate', optimizer.lr, epoch)

        print('\nLearning rate: {}'.format(optimizer.lr))
        print('Step num: {}\n'.format(optimizer.step_num))

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, train_loss)
    with open('loss.txt', 'w') as f:
        f.write('\n'.join(Loss_list))

def train(train_loader, model, optimizer, epoch, logger, writer):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    times = AverageMeter()

    start = time.time()

    # Batches
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(Config.device)
        padded_target = padded_target.to(Config.device)
        input_lengths = input_lengths.to(Config.device)

        # Forward prop.
        pred, gold = model(padded_input, input_lengths, padded_target)

        loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

        try:
            assert (not math.isnan(loss.item()))
        except AssertionError:
            print('n_correct: ' + str(n_correct))
            print('data: ' + str(n_correct))
            continue

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer.optimizer, Config.grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        elapsed = time.time() - start
        start = time.time()

        losses.update(loss.item())
        times.update(elapsed)

        # Print status
        if i % Config.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Batch time {time.val:.5f} ({time.avg:.5f})\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader), time=times,
                                                                      loss=losses))
            writer.add_scalar('step_num/train_loss', losses.avg, optimizer.step_num)
            writer.add_scalar('step_num/learning_rate', optimizer.lr, optimizer.step_num)

    return losses.avg


def main():
    global args
    args = parse_args()   # 解析命令行参数　并传入下面的网络中
    train_net(args)


if __name__ == '__main__':
    main()

