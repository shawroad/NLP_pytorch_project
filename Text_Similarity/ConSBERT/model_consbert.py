"""
@file   : model_esimcse.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-10-25
"""
import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class Model(nn.Module):
    def __init__(self, temperature=0.05, cutoff_rate=0.15, close_dropout=True):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./roberta_pretrain/config.json')
        self.bert = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.config)
     
        self.temperature = temperature
        self.cutoff_rate = cutoff_rate
        self.close_dropout = close_dropout
        self.loss_fct = nn.CrossEntropyLoss()
    
    def cal_cos_sim(self, embedding1, embedding2):
        embedding1_norm = F.normalize(embedding1, p=2, dim=1)
        embedding2_norm = F.normalize(embedding2, p=2, dim=1)
        return torch.mm(embedding1_norm, embedding2_norm.transpose(0, 1))  # (batch_size, batch_size)

    def shuffle_and_cutoff(self, input_ids, attention_mask):
        # input_ids, attention_mask = input_ids.cpu().numpy(), attention_mask.cpu().numpy()
        bsz, seq_len = input_ids.shape
        shuffled_input_ids = []
        cutoff_attention_mask = []
         
        for bsz_id in range(bsz):
            sample_mask = attention_mask[bsz_id]
            # num_tokens = sample_mask.sum().int().item()  # 当前句子中的token数
            num_tokens = sample_mask.sum().item()
            cur_input_ids = input_ids[bsz_id]   # 当前的input_ids
            if 102 not in cur_input_ids:
                indexes = list(range(num_tokens))[1:]
                random.shuffle(indexes)  # 打乱位置
                indexes = [0] + indexes  # 保证第一个位置是0
            else:
                indexes = list(range(num_tokens))[1:-1] 
                random.shuffle(indexes)
                indexes=[0] + indexes + [num_tokens-1]  # 保证第一个位置是0，最后一个位置是SEP不变
            rest_indexes = list(range(num_tokens, seq_len))
            total_indexes = indexes + rest_indexes
            shuffled_input_id=input_ids[bsz_id][total_indexes]  # 相当于是把token打乱了。
            # print(shuffled_input_id, indexes)

            if self.cutoff_rate > 0.0:
                sample_len = max(int(num_tokens * (1-self.cutoff_rate)), 1)  # if true_len is 32, cutoff_rate is 0.15 then sample_len is 27
                start_id = np.random.randint(1, high = num_tokens - sample_len + 1)  # start_id random select from (0,6)，避免删除CLS
                cutoff_mask = [1] * seq_len
                for idx in range(start_id, start_id+sample_len):
                    cutoff_mask[idx] = 0  # 这些位置是0，bool之后就变成了False，而masked_fill是选择True的位置替换为value的

                cutoff_mask[0] = 0  # 避免CLS被替换
                cutoff_mask[num_tokens - 1] = 0  # 避免SEP被替换
                cutoff_mask=torch.ByteTensor(cutoff_mask).bool().cuda()
                shuffled_input_id=shuffled_input_id.masked_fill(cutoff_mask,value=0).cuda()
                sample_mask=sample_mask.masked_fill(cutoff_mask,value=0).cuda()
            shuffled_input_id = shuffled_input_id.view(1, -1)
            sample_mask = sample_mask.view(1, -1)

            shuffled_input_ids.append(shuffled_input_id)
            cutoff_attention_mask.append(sample_mask)

        shuffled_input_ids = torch.cat(shuffled_input_ids, dim=0)
        cutoff_attention_mask = torch.cat(cutoff_attention_mask, dim=0)
        return shuffled_input_ids, cutoff_attention_mask

    def forward(self, input_ids1, attention_mask1):
        '''
        输入一个句子的input_ids 以及 attention_mask
        下面会采用esimcse的增广方式，生成其对应的正样本
        :param input_ids:
        :param attention_mask:
        :return:
        '''
        s1_embedding = self.bert(input_ids1, attention_mask1, output_hidden_states=True).last_hidden_state[:, 0]
        # print(s1_embedding.size())
        input_ids2, attention_mask2 = torch.clone(input_ids1), torch.clone(attention_mask1)
    
        shuffle_input_ids, cutoff_attention_mask = self.shuffle_and_cutoff(input_ids2, attention_mask2)
       
        orig_attention_probs_dropout_prob = self.config.attention_probs_dropout_prob
        orig_hidden_dropout_prob = self.config.hidden_dropout_prob
       
        if self.close_dropout:
            self.config.attention_probs_dropout_prob = 0.0
            self.config.hidden_dropout_prob = 0.0
        s2_embedding = self.bert(shuffle_input_ids, cutoff_attention_mask, output_hidden_states=True).last_hidden_state[:, 0]

        if self.close_dropout:
            self.config.attention_probs_dropout_prob = orig_attention_probs_dropout_prob
            self.config.hidden_dropout_prob = orig_hidden_dropout_prob
        
        cos_sim = self.cal_cos_sim(s1_embedding, s2_embedding) / self.temperature
          
        batch_size = cos_sim.size(0)
        assert cos_sim.size() == (batch_size, batch_size)
        labels = torch.arange(batch_size).cuda()
        loss = self.loss_fct(cos_sim, labels)
        return loss

    def encode(self, input_ids, attention_mask):
        s1_embedding = self.bert(input_ids, attention_mask, output_hidden_states=True).last_hidden_state[:, 0]
        return s1_embedding
  

