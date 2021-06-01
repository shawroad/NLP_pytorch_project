import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel
from config import set_args


args = set_args()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 加载预训练模型
        self.config = BertConfig.from_pretrained(args.pretrain_config)
        self.bert = BertModel.from_pretrained(args.pretrain_model, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True

        # self.lstm = nn.LSTM(768, 768//2, num_layers=3, batch_first=True, bidirectional=True)

        self.qa_outputs = nn.Linear(768, 2)
        self.loss_fct = CrossEntropyLoss()   # 计算损失

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        
        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            start_loss = self.loss_fct(start_logits, start_positions)
            end_loss = self.loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2  # !!!
            return total_loss, start_logits, end_logits

        else:
            return start_logits, end_logits