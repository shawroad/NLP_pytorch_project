"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-03-25
"""
import json
import time
import torch.optim
import torch.utils.data
from torch import nn
from config import set_args
import torchvision.transforms as transforms
from data_helper import CaptionDataset
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence
from model import Encoder, DecoderWithAttention
from utils import clip_gradient, AverageMeter, accuracy, save_checkpoint, adjust_learning_rate


def evaluate(val_loader, encoder, decoder, criterion):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    for step, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
        if torch.cuda.is_available():
            imgs, caps, caplens = imgs.cuda(), caps.cuda(), caplens.cuda()

        if encoder is not None:
            imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        targets = caps_sorted[:, 1:]

        scores_copy = scores.clone()
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        if torch.cuda.is_available():
            scores = scores.cuda()
            targets = targets.cuda()
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if step % args.print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                step, len(val_loader), batch_time=batch_time, loss=losses, top5=top5accs))

        allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    args = set_args()

    word_map = json.load(open('./data/WORDMAP.json', 'r', encoding='utf8'))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = CaptionDataset('train', transform=transforms.Compose([normalize]), word_map=word_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)   # num_workers=workers, pin_memory=True

    valid_dataset = CaptionDataset('valid', transform=transforms.Compose([normalize]), word_map=word_map)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5, pin_memory=True)

    encoder = Encoder()
    fine_tune_encoder = False
    encoder.fine_tune(fine_tune_encoder)
    decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                   embed_dim=args.emb_dim,
                                   decoder_dim=args.decoder_dim,
                                   vocab_size=len(word_map),
                                   dropout=args.dropout)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=args.decoder_lr)
    if fine_tune_encoder:
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.encoder_lr)
    else:
        encoder_optimizer = None
    

    criterion = nn.CrossEntropyLoss()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    best_bleu4 = 0.0
    epochs_since_improvement = 0
    start = time.time()

    for epoch in range(args.epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        encoder.train()
        decoder.train()
        for step, batch in enumerate(train_loader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)

            imgs, caps, caplens = batch

            imgs_out = encoder(imgs)
            # print(imgs_out.size())    # torch.Size([2, 14, 14, 2048])
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_out, caps, caplens)
            targets = caps_sorted[:, 1:]   # 解码器开始是<start> 所以计算损失 可以去掉

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
            if torch.cuda.is_available():
                scores = scores.cuda()
                targets = targets.cuda()

            loss = criterion(scores, targets)

            loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if fine_tune_encoder:
                encoder_optimizer.zero_grad()
            loss.backward()

            if args.grad_clip is not None:
                clip_gradient(decoder_optimizer, args.grad_clip)
                if fine_tune_encoder:
                    clip_gradient(encoder_optimizer, args.grad_clip)

            decoder_optimizer.step()
            if fine_tune_encoder:
                encoder_optimizer.step()
            start = time.time()
            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if step % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, step, len(train_loader),
                                                                              batch_time=batch_time,
                                                                              data_time=data_time, loss=losses,
                                                                              top5=top5accs))

        recent_bleu4 = evaluate(val_loader=valid_loader, encoder=encoder, decoder=decoder, criterion=criterion)
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
        else:
            epochs_since_improvement = 0
        save_checkpoint(epoch, epochs_since_improvement, encoder,
                        decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)

