# -*- coding: utf-8 -*-
# @Time    : 2020/9/17 9:36
# @Author  : xiaolu
# @FileName: train.py
# @Software: PyCharm
import os
import gzip
import torch
import random
import pickle
import time
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from model import Model
from config import set_args


class RankExample(object):
    def __init__(self,
                 qas_id,
                 question_text,
                 question_type,
                 doc_tokens,
                 doc_id,
                 answer=None,
                 label=None,
                 negative_doc_id=None,
                 negative_doc=None,
                 ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.question_type = question_type
        self.doc_id = doc_id
        self.doc_tokens = doc_tokens
        self.answer = answer
        self.label = label
        self.negative_doc_id = negative_doc_id
        self.negative_doc = negative_doc

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", question_type: %s" % (self.question_type)
        s += ", doc_id: %s" % (str(self.doc_id))
        s += ", doc_tokens: %s" % (self.doc_tokens)
        s += ", answer: %s" % (self.answer)
        s += ", label: %d" % (self.label)
        s += ", negative_doc_id: {}".format(self.negative_doc_id)
        s += ", negative_doc: {}".format(self.negative_doc)
        return s


class InputFeatures(object):
    def __init__(self, question_input_ids=None, question_input_mask=None, question_segment_ids=None,
                 pos_id=None, neg_id_list=None, context_input_ids=None, context_input_mask=None,
                 context_segment_ids=None):
        self.question_input_ids = question_input_ids
        self.question_input_mask = question_input_mask
        self.question_segment_ids = question_segment_ids
        self.pos_id = pos_id
        self.neg_id_list = neg_id_list

        self.context_input_ids = context_input_ids
        self.context_input_mask = context_input_mask
        self.context_segment_ids = context_segment_ids

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "  \n question_input_ids: {}".format(self.question_input_ids)
        s += ", \n question_input_mask: {}".format(self.question_input_mask)
        s += ", \n question_segment_ids: {}".format(self.question_segment_ids)
        s += ", \n pos_id: {}".format(self.pos_id)
        s += ", \n neg_id_list: {}".format(self.neg_id_list)
        s += ", \n context_input_ids: {}".format(self.context_input_ids)
        s += ", \n context_input_mask: {}".format(self.context_input_mask)
        s += ", \n context_segment_ids: {}".format(self.context_segment_ids)
        return s


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def dot_product_scores(q_vectors, ctx_vectors):
    # 计算相似度 这里可以换其他相似度计算函数
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))   # batch_size x *
    return r


class BiEncoderNllLoss(object):
    def calc(self, q_vectors, ctx_vectors, positive_idx_per_question: list, hard_negative_idx_per_question: list = None):
        '''
        :param q_vectors: torch.Size([2, 768])
        :param ctx_vectors: torch.Size([4, 768])
        :param positive_idx_per_question: [0, 2]
        :param hard_negative_idx_per_question: [[1], [3]]
        :return:
        '''
        scores = self.get_scores(q_vectors, ctx_vectors)
        # 其实很多余
        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)    # 问题数目
            scores = scores.view(q_num, -1)   #
        softmax_scores = F.log_softmax(scores, dim=1)
        loss = F.nll_loss(softmax_scores, torch.tensor(positive_idx_per_question).to(softmax_scores.device), reduction='mean')
        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector, ctx_vectors):
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


def _calc_loss(loss_function, local_q_vector, local_ctx_vectors, local_positive_idxs, local_hard_negatives_idxs):
    loss, is_correct = loss_function.calc(local_q_vector, local_ctx_vectors, local_positive_idxs, local_hard_negatives_idxs)
    return loss, is_correct


if __name__ == '__main__':
    args = set_args()
    set_seed(args)  # 设定随机种子

    device = torch.device('cuda: {}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    # 加载训练集
    tokenizer = BertTokenizer.from_pretrained('./roberta_pretrain/vocab.txt')
    with gzip.open('./data/features.pkl.gz', 'rb') as f:
        train_features = pickle.load(f)

    # Prepare Optimizer
    num_train_steps = int(
        len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # 模型
    model = Model()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = 0.05 * num_train_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    best_loss = None
    global_step = 0
    model.to(device)

    loss_function = BiEncoderNllLoss()

    if args.do_train:
        print("***** Running training *****")
        print("  Num examples = {}".format(len(train_features)))
        print("  Batch size = {}".format(args.train_batch_size))
        model.train()
        global_step = 0
        for epoch in range(args.num_train_epochs):
            STEP = len(train_features) // args.train_batch_size
            for step in range(STEP-1):
                start_time = time.time()
                batch = train_features[step * args.train_batch_size: (step + 1) * args.train_batch_size]
                # 构造输入
                question_input_ids_list = []
                question_input_mask_list = []
                question_segment_ids_list = []

                context_input_ids_list = []
                context_input_mask_list = []
                context_segment_ids_list = []

                pos_ids = []
                neg_ids_list = []
                for i, b in enumerate(batch):
                    question_input_ids_list.append(b.question_input_ids)
                    question_input_mask_list.append(b.question_input_mask)
                    question_segment_ids_list.append(b.question_segment_ids)

                    context_input_ids_list.extend(b.context_input_ids)
                    context_input_mask_list.extend(b.context_input_mask)
                    context_segment_ids_list.extend(b.context_segment_ids)
                    pos_ids.append(10*i + b.pos_id)
                    temp = []
                    for j in b.neg_id_list:
                        temp.append(10*i+j)
                    neg_ids_list.append(temp)

                question_input_ids = torch.LongTensor(question_input_ids_list).to(device)
                question_input_mask = torch.LongTensor(question_input_mask_list).to(device)
                question_segment_ids = torch.LongTensor(question_segment_ids_list).to(device)

                context_input_ids = torch.LongTensor(context_input_ids_list).to(device)
                context_input_mask = torch.LongTensor(context_input_mask_list).to(device)
                context_segment_ids = torch.LongTensor(context_segment_ids_list).to(device)
                # print(question_input_ids.size())    # torch.Size([2, 60])
                # print(question_input_mask.size())   # torch.Size([2, 60])
                # print(question_segment_ids.size())  # torch.Size([2, 60])
                # print(context_input_ids.size())     # torch.Size([20, 512])
                # print(context_input_mask.size())    # torch.Size([20, 512])
                # print(context_segment_ids.size())   # torch.Size([20, 512])
                # print(neg_ids_list)   # [[0, 1, 2, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 17, 18, 19]]
                # print(pos_ids)    # [3, 16]

                q_encode, c_encode = model(question_input_ids, question_input_mask, question_segment_ids,
                                           context_input_ids, context_input_mask, context_segment_ids)
                loss, is_correct = loss_function.calc(q_encode, c_encode, pos_ids, neg_ids_list)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                print('epoch:{}, step:{}, loss:{:10f}, time_cost:{:10f}'.format(epoch, step, loss, time.time()-start_time))
                loss.backward()

                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # 一个epoch跑完 保存模型
            os.makedirs(args.output_dir, exist_ok=True)
            # 总体模型保存
            q_c_encoder_file = os.path.join(args.output_dir, 'q_c_encode_epoch{}.bin'.format(epoch))
            torch.save(model.state_dict(), q_c_encoder_file)
