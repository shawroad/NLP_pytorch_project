"""

@file  : utils.py

@author: xiaolu

@time  : 2020-03-02

"""
from collections import Counter   # 统计列表中的词频


def precision_recall_f1(prediction, ground_truth):
    '''
    This function calculates and returns the precision, recall and f1-score
    :param prediction: prediction string or list to be matched
    :param ground_truth: golden string or list reference
    :return: floats of (p, r, f1)
    '''
    if not isinstance(prediction, list):
        prediction_tokens = prediction.split()
    else:
        prediction_tokens = prediction

    if not isinstance(ground_truth, list):
        ground_truth_tokens = ground_truth.split()
    else:
        ground_truth_tokens = ground_truth

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)  # 共同出现的词

    num_same = sum(common.values())  # 共现的词的个数
    if num_same == 0:
        # 没有共同出现的词
        return 0, 0, 0

    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def f1_score(prediction, ground_truth):
    '''
    This function calculates and returns the f1-score
    :param prediction: prediction string or list to be matched 某篇文章中的一段话
    :param ground_truth: golden string or list reference  问题
    :return: floats of f1
    '''
    return precision_recall_f1(prediction, ground_truth)[2]


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    # related_score = metric_max_over_ground_truths(f1_score, para_tokens, [question])
    '''
    This function calculates and returns the precision, recall and f1-score
    :param metric_fn: metric function pointer which calculates scores according to corresponding logic.
    :param prediction: prediction string or list to be matched
    :param ground_truths: golden string or list reference
    :return: floats of (p, r, f1)
    '''
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        # print(ground_truth)  # 问题的分词形式
        # 遍历问题中的每个词
        score = metric_fn(prediction, ground_truth)  # 计算当前句子与问题的f1_score
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
