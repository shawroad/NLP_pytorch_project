"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-07
"""
import torch
from sys import platform
from torch import nn
from config import set_args
args = set_args()
max_length = args.max_char_len


class Model(nn.Module):
    def __init__(self, embeddings, hihdden_size=300, dropout=0.2, num_classes=2):
        super(Model, self).__init__()
        self.embedding_dim = embeddings.shape[1]
        self.hidden_size = hihdden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.word_embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(embeddings))
        self.word_embedding.float()
        self.word_embedding.weight.requires_grad = True
        if self.dropout:
            self.rnn_dropout = RNNDropout(p=self.dropout)
        self.first_rnn = Seq2SeqEncoder(nn.LSTM, self.embedding_dim, self.hidden_size, bidirectional=True)
        self.projection = nn.Sequential(nn.Linear(4*2*self.hidden_size, self.hidden_size),
                                        nn.ReLU())
        self.attention = SoftmaxAttention()
        self.second_rnn = Seq2SeqEncoder(nn.LSTM, self.hidden_size, self.hidden_size, bidirectional=True)
        self.classification = nn.Sequential(nn.Linear(2*4*self.hidden_size, self.hidden_size),
                                            nn.ReLU(),
                                            nn.Dropout(p=self.dropout),
                                            nn.Linear(self.hidden_size, self.hidden_size//2),
                                            nn.ReLU(),
                                            nn.Dropout(p=self.dropout),
                                            nn.Linear(self.hidden_size//2, self.num_classes))

    def forward(self, q1, q1_lengths, q2, q2_lengths):
        q1_mask = get_mask(q1, q1_lengths)
        q2_mask = get_mask(q2, q2_lengths)
        q1_embed = self.word_embedding(q1)
        q2_embed = self.word_embedding(q2)
        if self.dropout:
            q1_embed = self.rnn_dropout(q1_embed)
            q2_embed = self.rnn_dropout(q2_embed)
        # 双向lstm编码
        q1_encoded = self.first_rnn(q1_embed, q1_lengths)
        q2_encoded = self.first_rnn(q2_embed, q2_lengths)
        # atention
        q1_aligned, q2_aligned = self.attention(q1_encoded, q1_mask, q2_encoded, q2_mask)
        # concat
        q1_combined = torch.cat([q1_encoded, q1_aligned, q1_encoded - q1_aligned, q1_encoded * q1_aligned], dim=-1)
        q2_combined = torch.cat([q2_encoded, q2_aligned, q2_encoded - q2_aligned, q2_encoded * q2_aligned], dim=-1)
        # 映射一下
        projected_q1 = self.projection(q1_combined)
        projected_q2 = self.projection(q2_combined)
        if self.dropout:
            projected_q1 = self.rnn_dropout(projected_q1)
            projected_q2 = self.rnn_dropout(projected_q2)
        # 再次经过双向RNN
        q1_compare = self.second_rnn(projected_q1, q1_lengths)
        q2_compare = self.second_rnn(projected_q2, q2_lengths)
        # 平均池化 + 最大池化
        q1_avg_pool = torch.sum(q1_compare * q1_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q1_mask, dim=1, keepdim=True)
        q2_avg_pool = torch.sum(q2_compare * q2_mask.unsqueeze(1).transpose(2, 1), dim=1)/torch.sum(q2_mask, dim=1, keepdim=True)
        q1_max_pool, _ = replace_masked(q1_compare, q1_mask, -1e7).max(dim=1)
        q2_max_pool, _ = replace_masked(q2_compare, q2_mask, -1e7).max(dim=1)
        # 拼接成最后的特征向量
        merged = torch.cat([q1_avg_pool, q1_max_pool, q2_avg_pool, q2_max_pool], dim=1)
        # 分类
        logits = self.classification(merged)
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities


class RNNDropout(nn.Dropout):
    def forward(self, sequences_batch):
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = torch.dropout(ones, self.p, self.training)
        return dropout_mask.unsqueeze(1) * sequences_batch


class Seq2SeqEncoder(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, bidirectional=False):
        "rnn_type must be a class inheriting from torch.nn.RNNBase"
        assert issubclass(rnn_type, nn.RNNBase)
        super(Seq2SeqEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.encoder = rnn_type(input_size, hidden_size, num_layers, bias=bias,
                                batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
        outputs, _ = self.encoder(packed_batch, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # for linux
        if platform == "linux" or platform == "linux2":
            reordered_outputs = outputs.index_select(0, restoration_idx)
        # for win10
        else:
            reordered_outputs = outputs.index_select(0, restoration_idx.long())
        return reordered_outputs


class SoftmaxAttention(nn.Module):
    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())
        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask)
        return attended_premises, attended_hypotheses


def get_mask(sequences_batch, sequences_lengths):
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones([batch_size, max_length], dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask


def replace_masked(tensor, mask, value):
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


def masked_softmax(tensor, mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = torch.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()
    return weighted_sum * mask


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)
    _, revese_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, revese_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index
