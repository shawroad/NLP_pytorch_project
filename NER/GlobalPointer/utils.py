"""
@file   : utils.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-04-06
"""
import torch
import numpy as np


class Preprocessor(object):
    def __init__(self, tokenizer, add_special_tokens=True):
        super(Preprocessor, self).__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def get_ent2token_spans(self, text, entity_list):
        """实体列表转为token_spans
        Args:
            text (str): 原始文本
            entity_list (list): [(start, end, ent_type),(start, end, ent_type)...]
        """
        ent2token_spans = []
        # eg: text=见面就要说say hello yesterday
        inputs = self.tokenizer(text, add_special_tokens=self.add_special_tokens, return_offsets_mapping=True)
        '''
        {'input_ids': [101, 6224, 7481, 2218, 6206, 6432, 10114, 8701, 9719, 8457, 8758, 102], 
        'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        'offset_mapping': [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 8), (9, 14), (15, 18), (18, 21), (21, 24), (0, 0)]}
        '''
        token2char_span_mapping = inputs["offset_mapping"]  # 每个切分后的token在原始的text中的起始位置和结束位置

        text2tokens = self.tokenizer.tokenize(text, add_special_tokens=self.add_special_tokens)
        # print(text2tokens)  #  ['[CLS]', '见', '面', '就', '要', '说', 'say', 'hello', 'yes', '##ter', '##day', '[SEP]']

        for ent_span in entity_list:
            ent = text[ent_span[0]: ent_span[1] + 1]    # 标注的实体位置就是按字符个数来数的
            ent2token = self.tokenizer.tokenize(ent, add_special_tokens=False)  # 对实体切分

            # 然后将按字符个数标注的位置  修订 成 分完词 以token为个体的位置
            token_start_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[0]]  # 可能会有多个
            token_end_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[-1]]  # 可能会有多个

            # 分词后的位置 转为字符寻址 要和之前标的地址要一致 否则 就出错了
            token_start_index = list(filter(lambda x: token2char_span_mapping[x][0] == ent_span[0], token_start_indexs))
            token_end_index = list(filter(lambda x: token2char_span_mapping[x][-1] - 1 == ent_span[1], token_end_indexs))
            # 上述 token2char_span_mapping[x][-1]-1 减1是因为原始的char_span是闭区间，而token2char_span是开区间

            if len(token_start_index) == 0 or len(token_end_index) == 0:
                # 无法对应的token_span中
                continue
            token_span = (token_start_index[0], token_end_index[0], ent_span[2])
            ent2token_spans.append(token_span)
        return ent2token_spans


def multilabel_categorical_crossentropy(y_pred, y_true):
    # 负例: (1 - 2 * 0) * y_pred = y_pred  正例: (1 - 2 * 1) * y_pred = - y_pred
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        
        try:
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        except:
            f1, precision, recall = 0, 0, 0
        return f1, precision, recall
