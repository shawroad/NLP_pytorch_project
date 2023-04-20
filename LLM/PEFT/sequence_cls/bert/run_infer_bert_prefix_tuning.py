"""
@file   : run_infer_bert_prefix_tuning.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2023-04-20
"""
import torch
import pandas as pd
from tqdm import tqdm
from config import set_args
from sklearn import metrics
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from data_helper import CustomDataset, collate_fn
from transformers.models.bert import BertForSequenceClassification, BertTokenizer


if __name__ == '__main__':
    args = set_args()
    args.output_dir = './prefix_tuning'
    val_df = pd.read_csv(args.val_data_path)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    val_dataset = CustomDataset(dataframe=val_df, tokenizer=tokenizer)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=args.val_batch_size,
                                collate_fn=collate_fn)

    config = PeftConfig.from_pretrained(args.output_dir)
    inference_model = BertForSequenceClassification.from_pretrained(args.pretrained_model_path)

    inference_model = PeftModel.from_pretrained(inference_model, args.output_dir)

    if torch.cuda.is_available():
        inference_model.cuda()

    inference_model.eval()
    eval_targets = []
    eval_predict = []
    for batch in tqdm(val_dataloader):
        if torch.cuda.is_available():
            batch = (t.cuda() for t in batch)
        input_ids, attention_mask, token_type_ids, label_ids = batch
        with torch.no_grad():
            outputs = inference_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits

        eval_targets.extend(label_ids.cpu().detach().numpy().tolist())
        eval_predict.extend(torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist())
        print(eval_targets)
        print(eval_predict)
    eval_accuracy = metrics.accuracy_score(eval_targets, eval_predict)
    print(eval_accuracy)

