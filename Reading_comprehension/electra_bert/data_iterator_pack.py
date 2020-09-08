# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 16:30
# @Author  : xiaolu
# @FileName: data_iterator_pack.py
# @Software: PyCharm

import torch
import numpy as np
from numpy.random import shuffle

IGNORE_INDEX = -100


class DataIteratorPack(object):
    def __init__(self, features, example_dict, bsz, device, sent_limit, entity_limit,
                 entity_type_dict=None, sequential=False, ):
        self.bsz = bsz  # batch_size
        self.device = torch.device('cuda: 7' if torch.cuda.is_available() else 'cpu')
        self.features = features
        self.example_dict = example_dict
        # self.entity_type_dict = entity_type_dict
        self.sequential = sequential
        self.sent_limit = sent_limit  # 80
        # self.para_limit = 4
        # self.entity_limit = entity_limit
        self.example_ptr = 0
        if not sequential:
            shuffle(self.features)

    def refresh(self):
        self.example_ptr = 0
        if not self.sequential:
            shuffle(self.features)

    def empty(self):
        return self.example_ptr >= len(self.features)

    def __len__(self):
        return int(np.ceil(len(self.features) / self.bsz))

    def __iter__(self):
        # BERT input
        context_idxs = torch.LongTensor(self.bsz, 512)
        context_mask = torch.LongTensor(self.bsz, 512)
        segment_idxs = torch.LongTensor(self.bsz, 512)

        query_mapping = torch.Tensor(self.bsz, 512).to(self.device)  # [1, 1, 1, 1, 1, 0, 0, 0, 0, 0...]用1表示出问题在的地方
        start_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).to(self.device)  # 句子的个数限制  第三维度的512可以理解为句向量
        all_mapping = torch.Tensor(self.bsz, 512, self.sent_limit).to(self.device)

        # Label tensor
        y1 = torch.LongTensor(self.bsz).to(self.device)  # 答案的起始位置
        y2 = torch.LongTensor(self.bsz).to(self.device)  # 答案的结束位置
        q_type = torch.LongTensor(self.bsz).to(self.device)  # 四分类
        is_support = torch.FloatTensor(self.bsz, self.sent_limit).to(self.device)  #

        while True:
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr
            cur_bsz = min(self.bsz, len(self.features) - start_id)
            cur_batch = self.features[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)  # 从长到短排序

            ids = []
            max_sent_cnt = 0
            max_entity_cnt = 0
            # start_mapping: batch_size,  sent_limit, 512
            # all_mapping: batch_size, 512, sent_limit
            # query_mapping: batch_size, 512
            for mapping in [start_mapping, all_mapping, query_mapping]:
                mapping.zero_()  # 将三者全部清空 用零进行填充

            is_support.fill_(0)  # is_support 也用零进行填充

            for i in range(len(cur_batch)):
                # 制作当前一个batch的数据
                case = cur_batch[i]  # 取出当前batch中的第一条数据
                # print(f'all_doc_tokens is {case.doc_tokens}')  # 下面为roberta的三个输入
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
                segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))

                for j in range(case.sent_spans[0][0] - 1):  # 取得是第一个句子的起始  将之前全部填充1  相当于就是将query全部搞成1 其他位置全部为零
                    query_mapping[i, j] = 1  #

                if case.ans_type == 0:  # 说明有答案 即有起始和结束位置
                    if len(case.end_position) == 0:  # 结束位置标记为零 相当于起始和终止
                        y1[i] = y2[i] = 0
                    elif case.end_position[0] < 512:  # 答案的结束位置不能超过512
                        y1[i] = case.start_position[0]
                        y2[i] = case.end_position[0]
                    else:
                        y1[i] = y2[i] = 0
                    q_type[i] = 0
                elif case.ans_type == 1:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1
                elif case.ans_type == 2:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2
                elif case.ans_type == 3:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 3

                # 限制句子不能超过sent_limit
                for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):
                    is_sp_flag = j in case.sup_fact_ids  # 表示当前这句话是否是支持句
                    start, end = sent_span  # 每个句子的开始和结束
                    if start < end:
                        is_support[i, j] = int(is_sp_flag)  # [0, 0, 1, 0, 1, 0, 0, 0...] 长度为sent_limit  1代表当前句子是相关句子
                        all_mapping[i, start:end + 1, j] = 1
                        '''
                        [[0, 0, 0, 0, 0, 0
                          0, 0, 0, 0, 0, 0
                          1, 0, 0, 0, 0, 0
                          1, 0, 0, 0, 0, 0
                          1, 0, 0, 0, 0, 0
                          0, 1, 0, 0, 0, 0
                          0, 1, 0, 0, 0, 0
                          0, 0, 1, 0, 0, 0
                          0, 0, 0, 1, 0, 0
                          0, 0, 0, 1, 0, 0
                          0, 0, 0, 0, 1, 0
                          0, 0, 0, 0, 0, 1
                          0, 0, 0, 0, 0, 1], 
                         []
                         ]
                        '''
                        start_mapping[i, j, start] = 1
                        '''
                        [[[0, 0, 0, 1, 0, 0, 0, 0, ...长度512],
                          [0, 0, 0, 0, 0, 0, 0, 1, ...长度512],
                          [0, 0, 0, 0, 0, 0, 0, 0, ...1长度512]
                        ],

                         []]
                        '''

                ids.append(case.qas_id)
                max_sent_cnt = max(max_sent_cnt, len(case.sent_spans))

            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            self.example_ptr += cur_bsz

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
                'query_mapping': query_mapping[:cur_bsz, :max_c_len].contiguous(),
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'ids': ids,
                'q_type': q_type[:cur_bsz],
                'start_mapping': start_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],
                'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],

                'is_support': is_support[:cur_bsz, :max_sent_cnt].contiguous(),
            }
