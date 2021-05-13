"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-10
"""
from torch import nn
import torch



class Model(nn.Module):
    def __init__(self, embeddings, f_in_dim=200, f_hid_dim=200, f_out_dim=200,
                 dropout=0.2, embedd_dim=300, num_classes=2):
        super(Model, self).__init__()
        self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings).float(), requires_grad=True)

        self.project_embedd = nn.Linear(embedd_dim, f_in_dim)
        self.F = nn.Sequential(nn.Dropout(0.2),
                               nn.Linear(f_in_dim, f_hid_dim),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(f_hid_dim, f_out_dim),
                               nn.ReLU())
        self.G = nn.Sequential(nn.Dropout(0.2),
                               nn.Linear(2 * f_in_dim, f_hid_dim),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(f_hid_dim, f_out_dim),
                               nn.ReLU())
        self.H = nn.Sequential(nn.Dropout(0.2),
                               nn.Linear(2 * f_in_dim, f_hid_dim),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(f_hid_dim, f_out_dim),
                               nn.ReLU())
        self.last_layer = nn.Linear(f_out_dim, num_classes)

    def forward(self, q1, q1_lengths, q2, q2_lengths):
        q1_mask = generate_sent_masks(q1, q1_lengths)
        q2_mask = generate_sent_masks(q2, q2_lengths)
        q1_embed = self.embed(q1)
        q2_embed = self.embed(q2)

        # project_embedd编码
        q1_encoded = self.project_embedd(q1_embed)
        q2_encoded = self.project_embedd(q2_embed)
        # Attentd
        attend_out1 = self.F(q1_encoded)
        attend_out2 = self.F(q2_encoded)
        similarity_matrix = attend_out1.bmm(attend_out2.transpose(2, 1).contiguous())
        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, q2_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), q1_mask)
        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        q1_aligned = weighted_sum(q2_encoded, prem_hyp_attn, q1_mask)
        q2_aligned = weighted_sum(q1_encoded, hyp_prem_attn, q2_mask)

        # compare
        compare_i = torch.cat((q1_encoded, q1_aligned), dim=2)
        compare_j = torch.cat((q2_encoded, q2_aligned), dim=2)
        v1_i = self.G(compare_i)
        v2_j = self.G(compare_j)
        # Aggregate (3.3)
        v1_sum = torch.sum(v1_i, dim=1)
        v2_sum = torch.sum(v2_j, dim=1)
        output_tolast = self.H(torch.cat((v1_sum, v2_sum), dim = 1))
        logits = self.last_layer(output_tolast)
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities


def generate_sent_masks(enc_hiddens, source_lengths):
    enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks


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