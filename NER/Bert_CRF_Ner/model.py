"""

@file  : model.py

@author: xiaolu

@time  : 2020-05-25

"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig
import numpy as np
from sklearn.metrics import f1_score, classification_report
from config import Config
from crf import CRF


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 加载预训练模型
        num_tag = 7
        self.config = BertConfig.from_pretrained(Config.model_config_path)
        self.bert = BertModel.from_pretrained(Config.model_path, config=self.config)
        self.qa_outputs = nn.Linear(768, num_tag)   # 总共有7个标签
        self.crf = CRF(num_tag)
        self.loss_fct = CrossEntropyLoss()   # 计算损失

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.qa_outputs(sequence_output)
        return logits

    def loss_fn(self, bert_encode, output_mask, tags):
        loss = self.crf.negative_log_loss(bert_encode, output_mask, tags)
        return loss

    def predict(self, bert_encode, output_mask):
        predicts = self.crf.get_batch_best_path(bert_encode, output_mask)
        predicts = predicts.view(1, -1).squeeze()
        predicts = predicts[predicts != -1]
        return predicts

    def acc_f1(self, y_pred, y_true):
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true==y_pred).astype(int))
        acc = correct/y_pred.shape[0]
        return acc, f1

    def class_report(self, y_pred, y_true):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)



