import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from config import Config


class SimplePredictionLayer(nn.Module):
    def __init__(self, config):
        super(SimplePredictionLayer, self).__init__()
        self.input_dim = config.input_dim

        self.sp_linear = nn.Linear(self.input_dim, 1)
        self.start_linear = nn.Linear(self.input_dim, 1)
        self.end_linear = nn.Linear(self.input_dim, 1)

        self.type_linear = nn.Linear(self.input_dim, config.label_type_num)   # yes/no/ans

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        # (batch, 512, 512)
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        # triu 生成上三角矩阵，tril生成下三角矩阵，这个相当于生成了(512, 512)的矩阵表示开始-结束的位置，答案长度最长为15
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, input_state):
        query_mapping = batch['query_mapping'].to(Config.device)  # (batch, 512) 不一定是512，可能略小
        context_mask = batch['context_mask'].to(Config.device)  # bert里实际有输入的位置
        all_mapping = batch['all_mapping'].to(Config.device)   # (batch_size, 512, sent_limit) 每个句子的token对应为1

        # print(query_mapping)    # torch.Size([2, 512])
        # print(context_mask)   # bert输入的那部分mask
        # print(all_mapping)    # torch.Size([2, 512, 1])   # 标识句子token
        # print(context_mask)    # [[1, 1, 1, 1, 0, 0, 0], []]
        #
        # data = all_mapping.numpy().tolist()
        # import json
        # json.dump(data, open('sample.json', 'w', encoding='utf8'))
        # exit()
        #
        # print(query_mapping.size())
        # print(context_mask.size())
        # print(all_mapping.size())
        # exit()
        # start_logits = torch.LongTensor(self.start_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)).to(device)
        # end_logits = torch.LongTensor(self.end_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)).to(device)
        start_logits = self.start_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)   # 防止预测的位置是padding处
        end_logits = self.end_linear(input_state).squeeze(2) - 1e30 * (1 - context_mask)
        # print(start_logits.size())   # torch.Size([4, 512])
        # print(end_logits.size())   # torch.Size([4, 512])

        # input_state   size: batch_size, max_len, hidden_size
        # sp_state = torch.LongTensor(all_mapping.unsqueeze(3) * input_state.unsqueeze(2)).to(device)  # N x sent x 512 x 300
        sp_state = all_mapping.unsqueeze(3) * input_state.unsqueeze(2)    # 相当于把句子向量拿出来了
        # batch_size, 512, sent_limit, 1     *  batch_size, 512, 1, hidden_size
        # print(sp_state.size())   # torch.Size([4, 512, sent_limit, 768])
        sp_state = sp_state.max(1)[0]
        # print(sp_state.size())   # torch.Size([4, sent_limit, 768])  # 从512个向量中，看每个维度谁大就取谁的值

        sp_logits = self.sp_linear(sp_state)
        # print(sp_logits.size())  # torch.Size([4, sent_limit, 1])

        type_state = torch.max(input_state, dim=1)[0]   # batch_size, 512, hidden_size
        # print(type_state.size())   # torch.Size([4, 768])
        type_logits = self.type_linear(type_state)
        # print(type_logits.size())   # torch.Size([4, 4])

        # 找结束位置用的开始和结束位置概率之和
        # (batch, 512, 1) + (batch, 1, 512) -> (512, 512)
        # temp1 = start_logits[:, :, None]   # start_logits: torch.Size([4, 512]) -> torch.Size([4, 512, 1])
        # temp2 = end_logits[:, None]   # end_logits: torch.Size([4, 512]) -> torch.Size([4, 1, 512])

        outer = start_logits[:, :, None] + end_logits[:, None]
        # print(temp1.size())   # torch.Size([4, 512, 1)
        # print(temp2.size())   # torch.Size([4, 1, 512)
        # print(outer.size())   # torch.Size([4, 512, 512])
        # print(temp1)
        # print(temp2)
        # print(outer)

        outer_mask = self.get_output_mask(outer)
        # print(outer_mask.size())    # torch.Size([512, 512])

        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))

        if query_mapping is not None:   # 这个是query_mapping (batch, 512)
            outer = outer - 1e30 * query_mapping[:, :, None]    # 不允许预测query的内容  # query_mapping[:, :, None]: torch.Size([4, 512, 1])

        # 这两句相当于找到了outer中最大值的i和j坐标
        # print(outer.size())   # torch.Size([4, 512, 512])
        start_position = outer.max(dim=2)[0].max(dim=1)[1]
        end_position = outer.max(dim=1)[0].max(dim=1)[1]
        # print(start_logits.size())   # torch.Size([4, 402])
        # print(end_logits.size())    # torch.Size([4, 402])
        # print(type_logits.size())   # torch.Size([4, 4])
        # print(sp_logits.squeeze(2).size())   # torch.Size([4, 26])
        # print(start_position.size())   # torch.Size([4])
        # print(end_position.size())   # torch.Size([4])

        return start_logits, end_logits, type_logits, sp_logits.squeeze(2), start_position, end_position


class BertSupportNet(nn.Module):
    """
    joint train bert and graph fusion net
    """

    def __init__(self, config, encoder):
        super(BertSupportNet, self).__init__()
        self.encoder = encoder
        self.graph_fusion_net = SupportNet(config).to(Config.device)

    def forward(self, batch, debug=False):
        doc_ids, doc_mask, segment_ids = batch['context_idxs'].to(Config.device), batch['context_mask'].to(Config.device), batch['segment_idxs'].to(Config.device)
        # print(doc_ids.size())    # torch.Size([4, 512])
        # print(doc_mask.size())    # torch.Size([4, 512])
        # print(segment_ids.size())   # torch.Size([4, 512])

        # roberta不可以输入token_type_ids
        all_doc_encoder_layers = self.encoder(input_ids=doc_ids,
                                              token_type_ids=segment_ids,  # 可以注释
                                              attention_mask=doc_mask)[0]   # batch_size,  max_len, hidden_size
        print(all_doc_encoder_layers.size())

        exit()

        batch['context_encoding'] = all_doc_encoder_layers
        return self.graph_fusion_net(batch)


class SupportNet(nn.Module):
    """
    Packing Query Version
    """

    def __init__(self, config):
        super(SupportNet, self).__init__()
        self.config = config  # 就是args
        # self.n_layers = config.n_layers  # 2
        self.max_query_length = 50
        self.prediction_layer = SimplePredictionLayer(config).to(Config.device)

    def forward(self, batch, debug=False):
        context_encoding = batch['context_encoding'].to(Config.device)
        # print(context_encoding.size())   # torch.Size([2, 410, 768])

        predictions = self.prediction_layer(batch, context_encoding)

        start_logits, end_logits, type_logits, sp_logits, start_position, end_position = predictions

        return start_logits, end_logits, type_logits, sp_logits, start_position, end_position
