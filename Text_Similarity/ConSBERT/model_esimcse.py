"""
@file   : esimcse.py
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
     
        self.moco_config = BertConfig.from_pretrained('./roberta_pretrain/config.json')
        self.moco_config.hidden_dropout_prob = 0.0
        self.moco_config.attention_probs_dropout_prob = 0.0
        self.moco_bert = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.moco_config)
    
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

    def word_repetition(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.cpu().tolist(), attention_mask.cpu().tolist()
        bsz, seq_len = len(input_ids), len(input_ids[0])
        repetitied_input_ids = []
        repetitied_attention_mask = []
        rep_seq_len = seq_len
        for bsz_id in range(bsz):
            sample_mask = attention_mask[bsz_id]
            actual_len = sum(sample_mask)

            cur_input_id=input_ids[bsz_id]
            dup_len=random.randint(a=0, b=max(2, int(self.dup_rate * actual_len)))
            dup_word_index=random.sample(list(range(1,actual_len)), k=dup_len)
            
            r_input_id=[]
            r_attention_mask=[]
            for index,word_id in enumerate(cur_input_id):
                if index in dup_word_index:
                    r_input_id.append(word_id)
                    r_attention_mask.append(sample_mask[index])

                r_input_id.append(word_id)
                r_attention_mask.append(sample_mask[index])

            after_dup_len=len(r_input_id)
            #assert after_dup_len==actual_len+dup_len
            repetitied_input_ids.append(r_input_id)#+rest_input_ids)
            repetitied_attention_mask.append(r_attention_mask)#+rest_attention_mask)

            assert after_dup_len==dup_len+seq_len
            if after_dup_len>rep_seq_len:
                rep_seq_len=after_dup_len

        for i in range(bsz):
            after_dup_len=len(repetitied_input_ids[i])
            pad_len=rep_seq_len-after_dup_len
            repetitied_input_ids[i]+=[0]*pad_len
            repetitied_attention_mask[i]+=[0]*pad_len

        repetitied_input_ids=torch.LongTensor(repetitied_input_ids).cuda()
        repetitied_attention_mask=torch.LongTensor(repetitied_attention_mask).cuda()
        return repetitied_input_ids, repetitied_attention_mask

    def forward(self, input_ids1, attention_mask1):
        '''
        输入一个句子的input_ids 以及 attention_mask
        下面会采用consbert的增广方式，生成其对应的正样本
        :param input_ids:
        :param attention_mask:
        :param encoder_type: encoder_type:  "first-last-avg", "last-avg", "cls", "pooler(cls + dense)"
        :return:
        '''
        s1_embedding = self.bert(input_ids1, attention_mask1, output_hidden_states=True).last_hidden_state[:, 0]
        # print(s1_embedding.size())

        input_ids2, attention_mask2 = torch.clone(input_ids1), torch.clone(attention_mask1)
    
        input_ids2, attention_mask2 = self.word_repetition(input_ids2, attention_mask2)
        s2_embedding = self.bert(input_ids2, attention_mask2, output_hidden_states=True).last_hidden_state[:, 0]
        # print(s2_embedding.size())
        
        cos_sim = self.cal_cos_sim(s1_embedding, s2_embedding) / self.temperature   # (batch_size, batch_size)
       
        batch_size = cos_sim.size(0)
        assert cos_sim.size() == (batch_size, batch_size)
        
        labels = torch.arange(batch_size).cuda()
        negative_samples = None
        if len(self.q) > 0:
            # 取负例
            # negative_samples = torch.vstack(self.q[:self.q_size])   # (q_size, 768)
            negative_samples = torch.cat(self.q[:self.q_size], dim=0)
        
        if len(self.q) + batch_size >= self.q_size:
            # 超过队列的最大值  直接将之前的一些元素出队
            del self.q[:batch_size]
        print('此时q队列中的样本数: ', len(self.q))

        with torch.no_grad():
            # 将当前这个batch的向量加入的队列中
            self.q.append(self.moco_bert(input_ids1, attention_mask1, output_hidden_states=True).last_hidden_state[:, 0])
     
        if negative_samples is not None:
            batch_size += negative_samples.size(0)   # N + M
            cos_sim_with_neg = self.cal_cos_sim(s1_embedding, negative_samples) / self.temperature   # (N, M)
            cos_sim = torch.cat([cos_sim, cos_sim_with_neg], dim=1)   # (N, N + M)
        
        for encoder_param, moco_encoder_param in zip(self.bert.parameters(), self.moco_bert.parameters()):
            moco_encoder_param.data = self.gamma * moco_encoder_param.data + (1. - self.gamma) * encoder_param.data
       
        loss = self.loss_fct(cos_sim, labels)
        return loss

    def encode(self, input_ids, attention_mask):
        s1_embedding = self.bert(input_ids, attention_mask, output_hidden_states=True).last_hidden_state[:, 0]
        return s1_embedding
  
