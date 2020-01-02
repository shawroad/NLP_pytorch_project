"""

@file   : bert.py

@author : xiaolu

@time   : 2019-12-31

"""
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer


class Config:
    def __init__(self, dataset):
        # 三种数据集的路径
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'   # 验证集
        self.test_path = dataset + '/data/test.txt'   # 测试集

        # 标签
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]      # 类别

        # 模型保存的位置
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'   # 模型训练结果保存的位置

        # 指定设备
        self.device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000   # 超过1000batch 效果没提升  则提前结束
        self.num_classes = len(self.class_list)   # 类别数

        self.num_epochs = 3   # 训练结果epoch
        self.batch_size = 128
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.vocab_bert_path = './bert_pretrain/bert-base-chinese-vocab.txt'
        self.tokenizer = BertTokenizer.from_pretrained(self.vocab_bert_path)   # 加载词表 然后进行分词 转id

        self.model_bert_path = './bert_pretrain/bert-base-chinese.tar.gz'  # 训练好的模型　配置文件
        self.hidden_size = 768   # 隐层的维度


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.model_bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True    # bert参数进行微调

        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)

        print(_.size())   # torch.Size([1, 32, 768]) batch_size x max_len x hidden_size
        print(pooled.size())  # torch.Size([1, 768]) batch_size x hidden_size
        exit()


        out = self.fc(pooled)
        return out
