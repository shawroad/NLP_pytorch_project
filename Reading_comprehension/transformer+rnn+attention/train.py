"""

@file  : train.py

@author: xiaolu

@time  : 2020-04-15

"""
import json
import torch
from torch import nn
import random
import datetime
import time
import os
from model import Encoder, Decoder
from config import Config
from rlog import _log_normal, _log_warning, _log_info, _log_error, _log_toomuch, _log_bg_blue, _log_bg_pp, _log_fg_yl, \
    _log_fg_cy, _log_black, rainbow


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(Config.device)
    return loss, nTotal.item()


def train():
    # 1.数据集整理
    data = json.load(open(Config.train_data_path, 'r'))

    input_data = data['input_data']
    input_len = data['input_len']
    output_data = data['output_data']
    mask_data = data['mask']
    output_len = data['output_len']

    total_len = len(input_data)
    step = total_len // Config.batch_size

    # 词嵌入部分
    embedding = nn.Embedding(Config.vocab_size, Config.hidden_size, padding_idx=Config.PAD)

    # 2. 模型准备
    encoder = Encoder(embedding)
    attn_model = 'dot'
    decoder = Decoder(attn_model, embedding,)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=Config.learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=Config.learning_rate)

    for epoch in range(Config.num_epochs):
        for i in range(step-1):
            start_time = time.time()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_ids = torch.LongTensor(input_data[i * Config.batch_size: (i + 1) * Config.batch_size]).to(Config.device)
            inp_len = torch.LongTensor(input_len[i * Config.batch_size: (i + 1) * Config.batch_size]).to(Config.device)
            output_ids = torch.LongTensor(output_data[i * Config.batch_size: (i + 1) * Config.batch_size]).to(Config.device)
            mask = torch.BoolTensor(mask_data[i * Config.batch_size: (i + 1) * Config.batch_size]).to(Config.device)
            out_len = output_len[i * Config.batch_size: (i + 1) * Config.batch_size]

            max_ans_len = max(out_len)

            mask = mask.permute(1, 0)
            output_ids = output_ids.permute(1, 0)
            encoder_outputs, hidden = encoder(input_ids, inp_len)
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            decoder_hidden = hidden.unsqueeze(0)

            # 创建解码的初始输入　(为一个batch中的每条数创建SOS)
            decoder_input = torch.LongTensor([[Config.SOS for _ in range(Config.batch_size)]])
            decoder_input = decoder_input.to(Config.device)

            # Determine if we are using teacher forcing this iteration
            teacher_forcing_ratio = 0.3
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            loss = 0
            print_losses = []
            n_totals = 0
            if use_teacher_forcing:
                # 这种是解码的每步我们输入上一步的真实标签
                for t in range(max_ans_len):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    # print(decoder_output.size())  # torch.Size([2, 2672])
                    # print(decoder_hidden.size())   # torch.Size([1, 2, 512])

                    decoder_input = output_ids[t].view(1, -1)
                    # 计算损失
                    mask_loss, nTotal = maskNLLLoss(decoder_output, output_ids[t], mask[t])
                    # print('1', mask_loss)
                    loss += mask_loss
                    print_losses.append(mask_loss.item() * nTotal)
                    n_totals += nTotal
            else:
                # 这种是解码的每步输入是上一步的预测结果
                for t in range(max_ans_len):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )

                    _, topi = decoder_output.topk(1)
                    decoder_input = torch.LongTensor([[topi[i][0] for i in range(Config.batch_size)]])
                    decoder_input = decoder_input.to(Config.device)
                    # Calculate and accumulate loss
                    mask_loss, nTotal = maskNLLLoss(decoder_output, output_ids[t], mask[t])
                    # print('2', mask_loss)
                    loss += mask_loss
                    print_losses.append(mask_loss.item() * nTotal)
                    n_totals += nTotal

            # Perform backpropatation
            loss.backward()

            # 梯度裁剪
            _ = nn.utils.clip_grad_norm_(encoder.parameters(), Config.clip)
            _ = nn.utils.clip_grad_norm_(decoder.parameters(), Config.clip)

            # Adjust model weights
            encoder_optimizer.step()
            decoder_optimizer.step()
            avg_loss = sum(print_losses) / n_totals

            time_str = datetime.datetime.now().isoformat()
            log_str = 'time:{}, epoch:{}, step:{}, loss:{:5f}, spend_time:{:6f}'.format(time_str, epoch, i, avg_loss, time.time() - start_time)
            rainbow(log_str)

        if epoch % 1 == 0:
            save_path = './save_model/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': avg_loss,
                'embedding': embedding.state_dict()
            }, os.path.join(save_path, 'epoch{}_{}_model.tar'.format(epoch, 'checkpoint')))


if __name__ == '__main__':
    train()