"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-27
"""
import torch
from torch import nn
from config import set_args
import torch.nn.functional as F
from transformers import BertConfig, BertModel

args = set_args()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_labels = 2
        self.config = BertConfig.from_pretrained(args.pretrain_config)
        self.bert = BertModel.from_pretrained(args.pretrain_model, config=self.config)
        self.qa_outputs = nn.Linear(self.config.hidden_size, self.num_labels)
        self.global_output = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.alph = nn.ParameterList([nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # 预测答案的起始和结束
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 对是否有答案进行预测
        ans_logit = self.global_output(pooled_output)

        new_weight = F.softmax(torch.cat([self.alph[0], self.alph[1]]))
        outputs = (start_logits, end_logits, ans_logit, new_weight,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)

            # 起始和结束的损失
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if torch.cuda.is_available():
                is_impossible = torch.tensor([0 if stp > 0 else 1 for stp in start_positions], dtype=torch.long).cuda()
            else:
                is_impossible = torch.tensor([0 if stp > 0 else 1 for stp in start_positions], dtype=torch.long)

            # unknown的损失
            is_imps_loss = loss_fct(ans_logit.view(-1, 2), is_impossible.view(-1))

            total_loss = new_weight[0] * total_loss + new_weight[1] * is_imps_loss
            outputs = (total_loss,) + outputs
        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


