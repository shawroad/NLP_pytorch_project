"""
@file   : model3.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-06-24
"""
import torch
from torch import nn
from config import set_args
import torch.nn.functional as F
from transformers.models.bert import BertModel, BertConfig


args = set_args()


# BERT+TextCNN
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # bert部分
        self.config = BertConfig.from_pretrained(args.bert_config)
        self.bert = BertModel.from_pretrained(args.bert_pretrain, config=self.config)
        self.dropout = nn.Dropout(0.5)

        # TextCNN部分
        filter_sizes = [2, 3, 4]   # 卷积核的尺寸
        filter_num = 128   # 卷积核的个数
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, kernel_size=(size, self.config.hidden_size))
             for size in filter_sizes]
        )

        mid_size = 256
        self.all_classifier = nn.Sequential(
            nn.Linear(len(filter_sizes) * filter_num, mid_size),
            nn.BatchNorm1d(mid_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mid_size, args.num_classes)
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_hidden = bert_output[0]   # torch.Size([2, 256, 768])

        output = self.dropout(bert_hidden)

        tcnn_input = output.unsqueeze(1)   # torch.Size([2, 1, 256, 768]  # 加了一个通道

        tcnn_output = [F.relu(conv(tcnn_input)).squeeze(3) for conv in self.convs]

        # print(tcnn_output[0].size())   # torch.Size([2, 128, 255])
        # print(tcnn_output[1].size())   # torch.Size([2, 128, 254])
        # print(tcnn_output[2].size())   # torch.Size([2, 128, 253])

        # 对三组数据分别进行池化然后拼接
        tcnn_output = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in tcnn_output]
        # print(tcnn_output[0].size())   # torch.Size([2, 128, 255])
        # print(tcnn_output[1].size())   # torch.Size([2, 128, 254])
        # print(tcnn_output[2].size())   # torch.Size([2, 128, 253])

        tcnn_output = torch.cat(tcnn_output, 1)
        tcnn_output = self.dropout(tcnn_output)

        all_probs = self.all_classifier(tcnn_output)
        return torch.sigmoid(all_probs)
