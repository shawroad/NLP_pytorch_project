import sys
# import ujson as json
import json
import re
import string
from collections import Counter
import pickle


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'unknown'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'unknown'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = list(normalized_prediction)
    ground_truth_tokens = list(normalized_ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall


def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


def eval(prediction_file, gold_file):
    # 预测文件夹  和写入文件夹
    with open(prediction_file, encoding='utf8') as f:
        prediction = json.load(f)
    with open(gold_file, encoding='utf8') as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    for dp in gold:
        cur_id = str(dp['_id'])
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)


if __name__ == '__main__':
    # eval(sys.argv[1], sys.argv[2])
    source_file = '../input/data.json'
    target_file = '../result/result.json'
    # eval(source_file, target_file)
    eval(target_file, source_file)
    # {'em': 0.7437699680511182, 'f1': 0.7813170200798731, 'prec': 0.7954309410858931, 'recall': 0.779036770762159, 'sp_em': 0.27284345047923325, 'sp_f1': 0.5059259412933504, 'sp_prec': 0.6249429484253755, 'sp_recall': 0.4731233835387194, 'joint_em': 0.2063897763578275, 'joint_f1': 0.4250667020625394, 'joint_prec': 0.5369160246923818, 'joint_recall': 0.3957428614163978}
    # 	Ans_F1	Sup_F1	Joint_F1: 70.49	 59.87	44.64
    # {'em': 0.8351437699680511, 'f1': 0.8821106717222645, 'prec': 0.8970892410748642, 'recall': 0.8808056590129077, 'sp_em': 0.7507987220447284, 'sp_f1': 0.7537509597253998, 'sp_prec': 0.773525026624068, 'sp_recall': 0.7523246614939901, 'joint_em': 0.6568690095846645, 'joint_f1': 0.6685766521649513, 'joint_prec': 0.6965695646206823, 'joint_recall': 0.6672187332948031}

    '''    
                              ans_em     ans_f1    sup_em   sup_f1   joint_em   joint_f1
    Baseline                  0.5944	0.6822      0.3571	0.5401    0.2704	0.3701
    Baseline  +  CAIL19 data  0.6658	0.7545      0.3929	0.5814    0.3163	0.4543
    '''


