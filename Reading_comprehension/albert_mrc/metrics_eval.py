"""

@file  : metrics_eval.py

@author: xiaolu

@time  : 2020-04-10

"""
import re
import nltk


def evaluate(ground_truth_file, prediction_file):
    f1 = 0
    em = 0
    total_count = 0
    for i, answers in enumerate(ground_truth_file):
        total_count += 1
        prediction = prediction_file[i]

        f1 += calc_f1_score(answers, prediction)
        em += calc_em_score(answers, prediction)

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score


def calc_f1_score(answers, prediction):

    #
    # for ans in answers:
    #     ans_segs = mixed_segmentation(ans, rm_punc=True)
    #
    #     prediction_segs = mixed_segmentation(prediction, rm_punc=True)
    #     lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
    #
    #     if lcs_len == 0:
    #         f1_scores.append(0)
    #         continue
    #     precision = 1.0 * lcs_len / len(prediction_segs)
    #     recall = 1.0 * lcs_len / len(ans_segs)
    #     f1 = (2 * precision * recall) / (precision + recall)
    #     f1_scores.append(f1)
    # return max(f1_scores)
    f1_scores = []
    ans_segs = mixed_segmentation(answers, rm_punc=True)
    prediction_segs = mixed_segmentation(prediction, rm_punc=True)
    if len(prediction_segs) == 0 or len(ans_segs):
        return 0
    lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
    precision = 1.0 * lcs_len / len(prediction_segs)
    recall = 1.0 * lcs_len / len(ans_segs)
    if prediction == 0 and recall == 0:
        return 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


def mixed_segmentation(in_str, rm_punc=False):
    # split Chinese with English
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def find_lcs(s1, s2):
    # find longest common string
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


def calc_em_score(answers, prediction):
    em = 0
    ans_ = remove_punctuation(answers)
    prediction_ = remove_punctuation(prediction)
    if ans_ == prediction_:
        em = 1
    return em


def remove_punctuation(in_str):
    # remove punctuation
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


if __name__ == '__main__':
    ground_truth_file = ['我啦啦啦啦', '草拟吗', '这次ok吗']
    prediction_file = ['我是你妈妈', '你报吗', '我非常好']
    F1, EM = evaluate(ground_truth_file, prediction_file)
    print(F1)   # 41.111111111111114
    print(EM)   # 0.0
