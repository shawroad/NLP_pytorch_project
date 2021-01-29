"""
@file   : model_concat.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-01-29
"""
import torch
from torch import nn
from transformers import BertModel, BertConfig
from torch.nn import CrossEntropyLoss

# 这个模型  我们是用roberta作为预训练模型 然后将每层的CLS的编码拿出来  计算一个加权和


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./roberta_pretrain/bert_config.json')
        self.roberta = BertModel.from_pretrained('./roberta_pretrain/pytorch_model.bin', config=self.config)
        self.num_labels = 2
        self.output = nn.Linear(self.config.hidden_size, self.num_labels)
        self.attn_linear = nn.Linear(self.config.hidden_size, 1)

    def get_attention_vec(self, all_layer_output):
        batch_size, max_len, hidden_size = all_layer_output[0].size()

        data_tensor = torch.empty(size=(len(all_layer_output), batch_size, hidden_size))
        if torch.cuda.is_available():
            data_tensor = data_tensor.cuda()

        for i, layer in enumerate(all_layer_output):
            data_tensor[i] = layer[:, 0, :]
        # print(data_tensor.size())   # torch.Size([13, 2, 768])

        x = torch.transpose(data_tensor, 0, 1)   # torch.Size([2, 13, 768])
        attn = self.attn_linear(x)   # torch.Size([2, 13, 1])

        x = torch.transpose(x, 1, 2)    # torch.Size([2, 768, 13])
        out = torch.matmul(x, attn)
        out = out.squeeze(-1)
        return out

    def forward(self, input_ids=None, attention_mask=None, segment_ids=None, labels=None):
        output = self.roberta(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask, output_hidden_states=True)
        # last_hidden_state = output['last_hidden_state']
        # pooler_output = output['pooler_output']
        all_layer_output = output['hidden_states']

        # 把所有层的cls位置拿出来  然后计算一个注意力
        out = self.get_attention_vec(all_layer_output)

        logits = self.output(out)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits

