"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-04-06
"""
import os
import torch.cuda
from tqdm import tqdm
from config import set_args
from model import GlobalPointer
from torch.utils.data import DataLoader
from data_helper import MyDataset, load_data, DataMaker
from transformers.models.bert import BertTokenizerFast
from utils import multilabel_categorical_crossentropy, MetricsCalculator
from transformers import AdamW, get_linear_schedule_with_warmup


def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


def evaluate(model, valid_dataloader):
    model.eval()
    total_f1, total_precision, total_recall = 0., 0., 0.
    for batch_data in tqdm(valid_dataloader, desc="Validating"):
        batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_data
        if torch.cuda.is_available():
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            batch_labels = batch_labels.cuda()
        with torch.no_grad():
            logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        sample_f1, sample_precision, sample_recall = metrics.get_evaluate_fpr(logits, batch_labels)
        total_f1 += sample_f1
        total_precision += sample_precision
        total_recall += sample_recall

    avg_f1 = total_f1 / (len(valid_dataloader))
    avg_precision = total_precision / (len(valid_dataloader))
    avg_recall = total_recall / (len(valid_dataloader))
    return avg_f1, avg_precision, avg_recall


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained('./roberta_pretrain', add_special_tokens=True, do_lower_case=False)
    ent2id_path = './data/ent2id.json'
    ent2id = load_data(ent2id_path, "ent2id")   # 实体类别个数
    ent_type_size = len(ent2id)

    data_maker = DataMaker(tokenizer)
    # 加载数据
    train_data = load_data(args.train_data_path, data_type="train")
    train_dataset = MyDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda x: data_maker.generate_batch(x, args.max_seq_len, ent2id))

    valid_data = load_data(args.valid_data_path, data_type='valid')
    valid_dataset = MyDataset(valid_data)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=lambda x: data_maker.generate_batch(x, args.max_seq_len, ent2id))

    total_steps = len(train_dataloader) * args.num_epochs

    model = GlobalPointer(ent_type_size)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps, num_training_steps=total_steps)

    metrics = MetricsCalculator()

    best_f1 = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch
            if torch.cuda.is_available():
                batch_input_ids = batch_input_ids.cuda()
                batch_attention_mask = batch_attention_mask.cuda()
                batch_token_type_ids = batch_token_type_ids.cuda()
                batch_labels = batch_labels.cuda()

            logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            # print(logits.size())    # torch.Size([2, 10, 512, 512])
            loss = loss_fun(logits, batch_labels)

            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            sample_precision = metrics.get_sample_precision(logits, batch_labels)
            sample_f1 = metrics.get_sample_f1(logits, batch_labels)
            print('epoch:{}, step:{}, loss:{:10f}, precision:{:10f}, f1:{:10f}'.format(epoch, step, loss.item(), sample_precision.item(), sample_f1.item()))
            
        # start evaluate / epoch
        avg_f1, avg_precision, avg_recall = evaluate(model, valid_dataloader)
        ss = 'epoch:{}, valid_f1:{:10f}, valid_precision:{:10f}, valid_recall:{:10f}'.format(epoch, avg_f1, avg_precision, avg_recall)
        print(ss)

        logs_path = os.path.join(args.output_dir, 'logs.txt')
        with open(logs_path, 'a+') as f:
            ss += '\n'
            f.write(ss)

        # 保存模型
        if avg_f1 > best_f1:
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "best_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "base_model_epoch_{}.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)


