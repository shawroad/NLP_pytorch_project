"""
@file   : model_simcse_GS_infoNCE.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-10-25
"""
import torch
import random
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class Model(nn.Module):
    def __init__(self, q_size=256, dup_rate=0.32, temperature=0.05, gamma=0.99):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./roberta_pretrain/config.json')
        self.bert = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.config)
        self.gamma = gamma
        self.q = []
        self.q_size = q_size
        self.dup_rate = dup_rate
        self.temperature = temperature
        self.loss_fct = nn.CrossEntropyLoss()
    
    def cal_cos_sim(self, embedding1, embedding2):
        embedding1_norm = F.normalize(embedding1, p=2, dim=1)
        embedding2_norm = F.normalize(embedding2, p=2, dim=1)
        return torch.mm(embedding1_norm, embedding2_norm.transpose(0, 1))  # (batch_size, batch_size)

    def forward(self, input_ids1, attention_mask1):
        '''
        :param input_ids:
        :param attention_mask:
        :param encoder_type: encoder_type:  "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
        :return:
        '''
        s1_embedding = self.bert(input_ids1, attention_mask1, output_hidden_states=True).last_hidden_state[:, 0]
        # print(s1_embedding.size())

        input_ids2, attention_mask2 = torch.clone(input_ids1), torch.clone(attention_mask1)
        s2_embedding = self.bert(input_ids2, attention_mask2, output_hidden_states=True).last_hidden_state[:, 0]
        # print(s2_embedding.size())
        
        cos_sim = self.cal_cos_sim(s1_embedding, s2_embedding) / self.temperature   # (batch_size, batch_size)
        mean, std = 0, 1
        reg_size = 32   # 随便定义  看给它增加多少负样本      
        hidden_size = 768 
        reg_random = torch.normal(mean, std, size=(reg_size, hidden_size)).cuda()
        # print(reg_random.size())  # torch.Size([32, 768])
        # print(s1_embedding.size())   # torch.Size([16, 768]) 
       
        reg_cos_sim = self.cal_cos_sim(s1_embedding, reg_random) / self.temperature
        # print(reg_cos_sim.size())  # torch.Size([16, 32]) 
        cos_sim = torch.cat((cos_sim, reg_cos_sim), dim=1)
        batch_size = cos_sim.size(0)
        labels = torch.arange(batch_size).cuda()
        loss = self.loss_fct(cos_sim, labels)
        return loss

    def encode(self, input_ids, attention_mask):
        s1_embedding = self.bert(input_ids, attention_mask, output_hidden_states=True).last_hidden_state[:, 0]
        return s1_embedding
  
