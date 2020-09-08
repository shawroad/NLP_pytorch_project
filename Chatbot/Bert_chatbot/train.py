"""

@file  : train.py

@author: xiaolu

@time  : 2020-03-25

"""
import torch
from torch.utils.data.dataloader import DataLoader
import time
import datetime
from seq2seq_bert import Seq2SeqModel
from bert_model import BertConfig
from dataloader import DreamDataset, collate_fn
from config import Config
from tokenizer import load_bert_vocab
from rlog import _log_normal, _log_warning, _log_info, _log_error, _log_toomuch, _log_bg_blue, _log_bg_pp, _log_fg_yl, _log_fg_cy, _log_black, rainbow


def load_model(model, pretrain_model_path):
    '''
    加载预训练模型
    :param model:
    :param pretrain_model_path:
    :return:
    '''
    checkpoint = torch.load(pretrain_model_path)
    # 模型刚开始训练的时候, 需要载入预训练的BERT
    checkpoint = {k[5:]: v for k, v in checkpoint.items()
                  if k[:4] == "bert" and "pooler" not in k}
    model.load_state_dict(checkpoint, strict=False)
    torch.cuda.empty_cache()
    print("bert预训练模型加载成功啦!!!")


def train():
    # 加载数据集
    dataset = DreamDataset()
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn)

    # 实例化模型
    word2idx = load_bert_vocab()
    bertconfig = BertConfig(vocab_size=len(word2idx))
    bert_model = Seq2SeqModel(config=bertconfig)
    # 加载预训练模型
    load_model(bert_model, Config.pretrain_model_path)
    bert_model.to(Config.device)

    # 声明需要优化的参数 并定义相关优化器
    optim_parameters = list(bert_model.parameters())
    optimizer = torch.optim.Adam(optim_parameters, lr=Config.learning_rate, weight_decay=1e-3)

    step = 0
    for epoch in range(Config.EPOCH):
        total_loss = 0
        i = 0
        for token_ids, token_type_ids, target_ids in dataloader:
            start_time = time.time()
            step += 1
            i += 1
            token_ids = token_ids.to(Config.device)
            token_type_ids = token_type_ids.to(Config.device)
            target_ids = target_ids.to(Config.device)
            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = bert_model(token_ids, token_type_ids, labels=target_ids, device=Config.device)

            # 1. 清空之前梯度
            optimizer.zero_grad()
            # 2. 反向传播
            loss.backward()
            # 3. 梯度更新
            optimizer.step()

            time_str = datetime.datetime.now().isoformat()

            log_str = 'time:{}, epoch:{}, step:{}, loss:{:8f}, spend_time:{:6f}'.format(time_str, epoch, step, loss, time.time() - start_time)
            rainbow(log_str)
            # print('epoch:{}, step:{}, loss:{:6f}, spend_time:{}'.format(epoch, step, loss, time.time() - start_time))

            # 为计算当前epoch的平均loss
            total_loss += loss.item()

            if step % 30 == 0:
                torch.save(bert_model.state_dict(), './bert_dream.bin')

        print("当前epoch:{}, 平均损失:{}".format(epoch, total_loss / i))

        if epoch % 10 == 0:
            save_path = "./data/" + "pytorch_bert_gen_epoch{}.bin".format(str(epoch))
            torch.save(bert_model.state_dict(), save_path)
            print("{} saved!".format(save_path))


if __name__ == '__main__':
    train()


