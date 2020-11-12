# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/11/12 14:00:50
@Author  :   xiaolu 
@Contact :   luxiaonlp@163.com
'''
import torch
from torch import nn
import torch.nn.functional as F
from mogrifier import Mogrifier
import math
from collections import namedtuple
from functools import partial
from inspect import isfunction

# structs

Memory = namedtuple('Memory', ['short', 'long'])

# helper functions

def to(t):
    return {'dtype': t.dtype, 'device': t.device}

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

def default(x, val):
    if x is not None:
        return x
    return val if not isfunction(val) else val()

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim+1] = split_dims
    return t.reshape(shape)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def queue_fifo(*args, length, dim=-2):
    queue = torch.cat(args, dim=dim)
    if length > 0:
        return split_at_index(dim, -length, queue)

    device = queue.device
    shape = list(queue.shape)
    shape[dim] = 0
    return queue, torch.empty(shape, device = device)

def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]

def iterate_tensor(t):
    length = t.shape[0]
    for ind in range(length):
        yield t[ind]

def init_parameter(shape, dim):
    t = torch.zeros(shape)
    std = 1 / math.sqrt(dim)
    t.uniform_(-std, std)
    return nn.Parameter(t)

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# neuromodulated bistable recurrent cell and other gating classes

class nBRC(nn.Module):
    def __init__(self, dims, hidden_dims):
        super().__init__()
        self.Ua = nn.Linear(dims, hidden_dims)
        self.Wa = nn.Linear(dims, hidden_dims)
        self.Uc = nn.Linear(dims, hidden_dims)
        self.Wc = nn.Linear(dims, hidden_dims)
        self.U  = nn.Linear(dims, hidden_dims)

    def forward(self, x, h):
        l = lambda linear, tensor: F.linear(tensor, linear.weight.clone(), linear.bias.clone())

        a = 1 + torch.tanh(l(self.Ua, x) + l(self.Wa, h))
        c = torch.sigmoid(l(self.Uc, x) + l(self.Wc, h))
        return c * h + (1 - c) * torch.tanh(l(self.U, x) + a * h)

class GRUGating(nn.Module):
    def __init__(self, dim, fn, mogrify = False):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.gru = nBRC(dim, dim)
        self.mogrify = Mogrifier(dim, factorize_k = dim // 4) if mogrify else None

    def forward(self, x, **kwargs):
        shape = x.shape
        dim = self.dim

        y = self.fn(x, **kwargs)

        if self.mogrify is not None:
            y, x = self.mogrify(y, x)

        gated_output = self.gru(
            y.reshape(-1, dim),
            x.reshape(-1, dim)
        )

        return gated_output.reshape(shape)

# feedforward

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

# attention.

class SelfAttention(nn.Module):
    def __init__(self, dim, seq_len, mem_len, lmem_len, heads = 8, attn_dropout = 0., dropout = 0., memory_attn_dropout = 0., one_kv_head = False, num_mem_kv = 4):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.heads = heads
        self.dim_head = dim // heads
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.lmem_len = lmem_len
        self.scale = self.dim_head ** (-0.5)

        self.to_q = nn.Linear(dim, dim, bias = False)

        kv_dim = self.dim_head if one_kv_head else dim
        self.to_kv = nn.Linear(dim, kv_dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.mem_kv = init_parameter((1, num_mem_kv, dim), dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

        self.memory_attn_dropout = nn.Dropout(memory_attn_dropout)

    def forward(self, x, memories = None, pos_emb = None, input_mask = None, calc_memory = True, **kwargs):
        b, t, e, h, dim_h = *x.shape, self.heads, self.dim_head

        memories = default(memories, (None, None))
        mem, lmem = memories

        init_mem = lambda: torch.empty(b, 0, e, **to(x))
        mem = default(mem, init_mem)
        lmem = default(lmem, init_mem)
        mem_kv = self.mem_kv.expand(b, -1, -1)

        mem_len, lmem_len, mem_kv_len = map(lambda t: t.shape[1], (mem, lmem, mem_kv))

        q = self.to_q(x)

        kv_input = torch.cat((mem_kv, lmem, mem, x), dim=1)
        kv_len = kv_input.shape[1]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        merge_heads = lambda x: reshape_dim(x, -1, (-1, dim_h)).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))

        k, v = map(lambda x: x.expand(-1, h, -1, -1), (k, v))

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if pos_emb is not None:
            pos_emb = pos_emb[:, -kv_len:].type(q.dtype)
            pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * self.scale
            pos_dots = shift(pos_dots)
            pos_dots = F.pad(pos_dots, (dots.shape[-1] - pos_dots.shape[-1], 0), value = 0.)
            dots = dots + pos_dots

        if input_mask is not None:
            mask = input_mask[:, None, :, None] * input_mask[:, None, None, :]
            mask = F.pad(mask, (mem_len + lmem_len + mem_kv_len, 0), value = True)
            dots.masked_fill_(~mask, mask_value)

        total_mem_len = mem_len + lmem_len + mem_kv_len
        mask = torch.ones(t, t + total_mem_len, **to(x)).triu_(diagonal = 1 + total_mem_len).bool()
        dots.masked_fill_(mask[None, None, ...], mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        out = self.to_out(out)

        return self.dropout(out)

# memory attention network

def linear_attn(q, k, v):
    q, k = q.softmax(dim=-1), k.softmax(dim=-2)
    context = torch.einsum('bhnd,bhne->bhde', k, v)
    out = torch.einsum('bhnd,bhde->bhne', q, context)
    return out

def full_attn(q, k, v):
    dots = torch.einsum('bhid,bhjd->bhij', q, k) * q.shape[-1] ** -0.5
    dots = dots.softmax(dim=-1)
    out = torch.einsum('bhij,bhjd->bhid', dots, v)
    return out

class LinearSelfAttention(nn.Module):
    def __init__(self, dim, depth, heads = 8):
        super().__init__()
        self.dim_head = dim // heads
        self.norm = nn.LayerNorm(dim, elementwise_affine = False)

        self.to_q = init_parameter((dim, dim), dim)
        self.to_kv = init_parameter((dim, 2 * dim), dim)
        self.to_out = init_parameter((dim, dim), dim)

    def forward(self, x, hiddens = None):
        dim_head = self.dim_head
        w_q, w_kv, w_out = map(torch.clone, (self.to_q, self.to_kv, self.to_out))
        
        normed_lmem = self.norm(x)
        q = torch.einsum('bnd,de->bne', normed_lmem, w_q)

        kv_input = torch.cat((normed_lmem, hiddens), dim=1)
        k, v = torch.einsum('bnd,de->bne', kv_input, w_kv).chunk(2, dim=-1)

        q, k, v = map(lambda t: reshape_dim(t, -1, (-1, dim_head)).transpose(-2, -3), (q, k, v))

        out = linear_attn(q, k, v)

        out = out.transpose(2, 3).reshape_as(x)
        out = torch.einsum('bnd,de->bne', out, w_out)
        return out

class MemoryAttentionNetwork(nn.Module):
    def __init__(self, dim, num_memory_depth, mem_len, lmem_len, heads = 4, num_attn_steps = 2, num_mem_kv = 4, mem_write_iters = 2):
        super().__init__()
        self.num_memory_depth = num_memory_depth
        self.mem_len = mem_len
        self.lmem_len = lmem_len

        self.dim = dim
        dim_head = dim // heads
        self.dim_head = dim_head

        self.depth_emb = init_parameter((num_memory_depth, 1, 1, 1), dim)
        self.init_lmem = init_parameter((1, 1, dim), dim)
        self.lmem_pos_emb = init_parameter((1, lmem_len, dim), dim)

        self.mem_kv = init_parameter((1, num_mem_kv, dim), dim)

        self.attn = LinearSelfAttention(dim, num_memory_depth, heads = heads)
        self.gate = nBRC(dim, dim)
        self.mem_write_iters = mem_write_iters

    def forward(self, lmem, smem, hiddens, detach_lmem = False):
        batch, dim, dim_head, mem_depth, lmem_len = lmem.shape[0], self.dim, self.dim_head, self.num_memory_depth, self.lmem_len

        # properly detach hidden state, and detach long term memory if truncate signal is given

        hiddens = hiddens.detach()

        if detach_lmem:
            lmem = lmem.detach()

        # initialize long term memory state if none provided

        if lmem is None or lmem.shape[1] == 0:
            lmem = self.init_lmem.clone().expand(batch, lmem_len, -1)

        # use efficient linear attention for updating long term memory

        next_lmem = lmem + self.lmem_pos_emb

        hiddens_and_smem = torch.cat((smem, hiddens), dim=-2)
        all_hiddens = (hiddens_and_smem + self.depth_emb).transpose(0, 1).reshape(batch, -1, dim)
        all_hiddens = torch.cat((all_hiddens, self.mem_kv.expand(batch, -1, -1)), dim=1)

        for _ in range(self.mem_write_iters):
            attn_out = self.attn(next_lmem, hiddens = all_hiddens)
            next_lmem = self.gate(attn_out, next_lmem)

        # fifo queue the short term memory
        _, next_mem = queue_fifo(smem, hiddens, length = self.mem_len, dim = 2)

        return Memory(short = next_mem.detach(), long = next_lmem)

# transformer
class MemoryTransformerXL(nn.Module):
    def __init__(self, num_tokens, dim, seq_len, depth, emb_dim = None, memory_layers = None, mem_len = None, lmem_len = None, heads = 8, gru_gated_residual = True, mogrify_gru = False, attn_dropout = 0., ff_glu = False, ff_dropout = 0., attn_layer_dropout = 0., one_kv_head = False, num_mem_kv = 0, mem_write_iters = 2):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        mem_len = default(mem_len, seq_len)
        lmem_len = default(lmem_len, mem_len)

        memory_layers = default(memory_layers, list(range(1, depth + 1)))

        assert all([layer > 0 and layer <= depth for layer in memory_layers]), 'one of the indicated memory layers is invalid'

        self.mem_len = mem_len
        self.seq_len = seq_len

        self.depth = depth
        self.memory_layers = list(memory_layers)

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.to_model_dim = nn.Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

        seq_and_mem_len = seq_len + mem_len + lmem_len
        self.pos_emb = nn.Parameter(torch.zeros(heads, seq_and_mem_len, dim // heads))
        
        self.to_logits = nn.Sequential(
            nn.Identity() if emb_dim == dim else nn.Linear(dim, emb_dim),
            # nn.Linear(emb_dim, num_tokens)
        )

        wrapper = partial(GRUGating, dim, mogrify = mogrify_gru) if gru_gated_residual else Residual

        self.attn_layers = nn.ModuleList([wrapper(PreNorm(dim, SelfAttention(dim, seq_len, mem_len, lmem_len, heads, dropout = attn_layer_dropout, attn_dropout = attn_dropout, one_kv_head = one_kv_head, num_mem_kv = num_mem_kv))) for _ in range(depth)])
        self.ff_layers = nn.ModuleList([wrapper(PreNorm(dim, FeedForward(dim, dropout = ff_dropout, glu = ff_glu))) for _ in range(depth)])

        self.memory_network = MemoryAttentionNetwork(dim, len(self.memory_layers), mem_len, lmem_len, num_mem_kv = num_mem_kv, mem_write_iters = mem_write_iters)

        # 将所有的token输出聚合  然后分类
        self.fc_attn = nn.Linear(dim, 1)
        self.outputs = nn.Linear(dim, 2)


    def forward(self, x, memories = None, mask = None, detach_lmem = False):
        x = self.token_emb(x)
        x = self.to_model_dim(x)
        b, t, d = x.shape

        assert t <= self.seq_len, f'input contains a sequence length {t} that is greater than the designated maximum sequence length {self.seq_len}'

        memories = default(memories, (None, None))
        mem, lmem = memories

        num_memory_layers = len(self.memory_layers)

        mem = default(mem, lambda: torch.empty(num_memory_layers, b, 0, d, **to(x)))
        lmem = default(lmem, lambda: torch.empty(b, 0, d, **to(x)))

        mem_len, lmem_len = map(lambda t: t.shape[2], (mem, lmem))
        total_len = mem_len + lmem_len + self.seq_len

        pos_emb = self.pos_emb[:, (self.seq_len - t):total_len]

        mem_iter = iterate_tensor(mem)

        hiddens = []

        for ind, (attn, ff) in enumerate(zip(self.attn_layers, self.ff_layers)):
            layer_num = ind + 1
            use_memory = layer_num in self.memory_layers
            memories = (next(mem_iter), lmem) if use_memory else None

            if use_memory:
                hiddens.append(x)

            x = attn(x, memories = memories, input_mask = mask, pos_emb = pos_emb)
            x = ff(x)

        hiddens = torch.stack(hiddens)
        out = self.to_logits(x)

        # my code
        alpha = F.softmax(self.fc_attn(out), dim=1)
        x = torch.sum(out * alpha, 1)
        x = F.relu(x)

        logits = self.outputs(x)
        # calculate next memory state
        # only push hidden to short term memory if input sequence length is full

        if t < self.mem_len:
            return out, Memory(short = mem, long = lmem)

        next_memory = self.memory_network(lmem, mem, hiddens, detach_lmem = detach_lmem)
        # return out, next_memory
        return logits
