# -*- coding: utf-8 -*-
# @Time    : 2020/7/9 14:07
# @Author  : xiaolu
# @FileName: dataloader.py
# @Software: PyCharm
import torch
import random

IGNORE_INDEX = -100


class DataIterator:
    def __init__(self, buckets, bsz, para_limit, ques_limit, char_limit, shuffle, sent_limit):
        self.buckets = buckets
        self.bsz = bsz
        if para_limit is not None and ques_limit is not None:
            self.para_limit = para_limit
            self.ques_limit = ques_limit
        else:
            para_limit, ques_limit = 0, 0
            for bucket in buckets:
                for dp in bucket:
                    para_limit = max(para_limit, dp['context_idxs'].size(0))
                    ques_limit = max(ques_limit, dp['ques_idxs'].size(0))
            self.para_limit, self.ques_limit = para_limit, ques_limit
        self.char_limit = char_limit
        self.sent_limit = sent_limit

        self.num_buckets = len(self.buckets)
        self.bkt_pool = [i for i in range(self.num_buckets) if len(self.buckets[i]) > 0]
        if shuffle:
            for i in range(self.num_buckets):
                random.shuffle(self.buckets[i])
        self.bkt_ptrs = [0 for i in range(self.num_buckets)]
        self.shuffle = shuffle

    def __iter__(self):
        context_idxs = torch.LongTensor(self.bsz, self.para_limit)
        ques_idxs = torch.LongTensor(self.bsz, self.ques_limit)
        context_char_idxs = torch.LongTensor(self.bsz, self.para_limit, self.char_limit)
        ques_char_idxs = torch.LongTensor(self.bsz, self.ques_limit, self.char_limit)
        y1 = torch.LongTensor(self.bsz)
        y2 = torch.LongTensor(self.bsz)
        q_type = torch.LongTensor(self.bsz)
        start_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit)
        end_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit)
        all_mapping = torch.Tensor(self.bsz, self.para_limit, self.sent_limit)
        is_support = torch.LongTensor(self.bsz, self.sent_limit)

        while True:
            if len(self.bkt_pool) == 0: break
            bkt_id = random.choice(self.bkt_pool) if self.shuffle else self.bkt_pool[0]
            start_id = self.bkt_ptrs[bkt_id]
            cur_bucket = self.buckets[bkt_id]
            cur_bsz = min(self.bsz, len(cur_bucket) - start_id)

            ids = []

            cur_batch = cur_bucket[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: (x['context_idxs'] > 0).long().sum(), reverse=True)

            max_sent_cnt = 0
            for mapping in [start_mapping, end_mapping, all_mapping]:
                mapping.zero_()
            is_support.fill_(IGNORE_INDEX)

            for i in range(len(cur_batch)):
                context_idxs[i].copy_(cur_batch[i]['context_idxs'])
                ques_idxs[i].copy_(cur_batch[i]['ques_idxs'])
                context_char_idxs[i].copy_(cur_batch[i]['context_char_idxs'])
                ques_char_idxs[i].copy_(cur_batch[i]['ques_char_idxs'])
                if cur_batch[i]['y1'] >= 0:
                    y1[i] = cur_batch[i]['y1']
                    y2[i] = cur_batch[i]['y2']
                    q_type[i] = 0
                elif cur_batch[i]['y1'] == -1:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1
                elif cur_batch[i]['y1'] == -2:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2
                elif cur_batch[i]['y1'] == -3:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 3
                else:
                    assert False
                ids.append(cur_batch[i]['id'])

                for j, cur_sp_dp in enumerate(cur_batch[i]['start_end_facts']):
                    if j >= self.sent_limit: break
                    if len(cur_sp_dp) == 3:
                        start, end, is_sp_flag = tuple(cur_sp_dp)
                    else:
                        start, end, is_sp_flag, is_gold = tuple(cur_sp_dp)
                    if start < end:
                        start_mapping[i, start, j] = 1
                        end_mapping[i, end-1, j] = 1
                        all_mapping[i, start:end, j] = 1
                        is_support[i, j] = int(is_sp_flag)

                max_sent_cnt = max(max_sent_cnt, len(cur_batch[i]['start_end_facts']))

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())
            max_q_len = int((ques_idxs[:cur_bsz] > 0).long().sum(dim=1).max())

            self.bkt_ptrs[bkt_id] += cur_bsz
            if self.bkt_ptrs[bkt_id] >= len(cur_bucket):
                self.bkt_pool.remove(bkt_id)

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'ques_idxs': ques_idxs[:cur_bsz, :max_q_len].contiguous(),
                   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'ques_char_idxs': ques_char_idxs[:cur_bsz, :max_q_len].contiguous(),
                   'context_lens': input_lengths,
                   'y1': y1[:cur_bsz],
                   'y2': y2[:cur_bsz],
                   'ids': ids,
                   'q_type': q_type[:cur_bsz],
                   'is_support': is_support[:cur_bsz, :max_sent_cnt].contiguous(),
                   'start_mapping': start_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                   'end_mapping': end_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                   'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt]}
