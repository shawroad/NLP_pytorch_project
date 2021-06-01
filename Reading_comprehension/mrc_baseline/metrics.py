"""
@file   : metrics.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-28
"""
import io
import json
import math
import re
import string
import collections
from transformers import BasicTokenizer


def _get_best_indexes(logits, n_best_size):
    '''

    :param logits: 预测的logits
    :param n_best_size: 取前几个最大
    :return:
    '''
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    '''
    将标记化预测投射回原始文本。
    :param pred_text: 预测的答案   在token中
    :param orig_text: 预测的答案   在doc_token中
    :param do_lower_case:
    :param verbose_logging:
    :return:
    For example: pred_text = steve smith; orig_text=steve smith's
    我们不想返回orig_text  因为这个答案中有个多余的's, 在pred_text和orig_text使用启发方法
    '''
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # 首先对原始进行tokenizer, 然后取出空格，检查原始和预测的答案长度是否一致
    # 如果不一致，启发式方法失败，否则，我们假设字符是一对一对齐的。
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)   # 默认在doc_token中得到的答案更全，在这个序列中找pred_text(在tokens中得到的答案)
    if start_position == -1:
        if verbose_logging:
            print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text

    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            print("Length not equal after stripping spaces: {} vs {}".format(orig_ns_text, tok_ns_text))
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position: (orig_end_position + 1)]
    return output_text


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def compute_predictions_logits(all_examples, all_features, all_results, n_best_size, max_answer_length,
                               do_lower_case, output_prediction_file, output_nbest_file, output_null_log_odds_file,
                               verbose_logging, version_2_with_negative, null_score_diff_threshold, tokenizer):
    '''
    :param all_examples: 所有的example
    :param all_features: 所有的feature
    :param all_results: 样本的id+起始预测的logit和结束预测的logits
    :param n_best_size: 每个样本预测前几个最好的答案
    :param max_answer_length: 预测答案的最大长度
    :param do_lower_case:
    :param output_prediction_file: 预测答案
    :param output_nbest_file: 前几个最好
    :param output_null_log_odds_file:
    :param verbose_logging:
    :param version_2_with_negative: 是否包含不可回答问题
    :param null_score_diff_threshold:
    :param tokenizer:
    :return:
    '''
    print(all_examples)
    print(len(all_examples))
    exit()

    print("Writing predictions to: %s" % (output_prediction_file))
    print("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:   # {'原始样本id1': [多个feature 因为滑动窗口], '原始样本id2': [多个feature 因为滑动窗口], ...}
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:    # {'现在样本id1': 答案, '现在样本id2': 答案, ...}
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for example_index, example in enumerate(all_examples):
        print(example_index)
        # 按原始样本遍历   先取出当前这个样本对应的几个features
        features = example_index_to_features[example_index]

        score_null = 1000000    # 答案为空的得分
        prelim_predictions = []   # 保存初步预测结果   考虑了起始和结束的多种组合
        min_null_feature_index = 0
        null_start_logit = 0
        null_end_logit = 0
        for feature_index, feature in enumerate(features):
            result = unique_id_to_result[feature.unique_id]  # 取出每一个feature的预测结果
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)  # 得到起始位置预测概率前几大的索引
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)  # 得到结束位置预测概率前几大的索引

            if version_2_with_negative:
                # 如果我们有不相关的答案，得到不相关的最小分数
                feature_null_score = result.start_logits[0] + result.end_logits[0]  # 取两种预测的CLS向量的概率值。

                # 选出当前样本中几个feature中无答案的最小得分  并保存feature的索引 以及起始和结束的概率值。
                if feature_null_score < score_null:
                    score_null = feature_null_score   # null最低分
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]   # 最低分的那个样本的起始位置的概率
                    null_end_logit = result.end_logits[0]    # 最低分的那个样本的结束位置的概率

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):   # 起始位置 超过本条样本的token数量
                        continue
                    if end_index >= len(feature.tokens):    # 结束位置 超过本条样本的token数量
                        continue
                    if start_index not in feature.token_to_orig_map:
                        # token_to_orig_map: {12:0, 13: 1, ...}  是文章id在input_ids的索引和原始文章的索引的字典
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        # 起始位置必须是本slice中文章的起始位置
                        continue

                    if end_index < start_index:
                        continue

                    length = end_index - start_index + 1
                    if length > max_answer_length:   # 答案长度不能超过指定长度
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )

        if version_2_with_negative:   # SquAD2.0 带有不可回答的问题
            # 加入预测为空的概率
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )

        # 按起始和结束的概率加和 然后降序排列
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        # 遍历prelim_predictions 找出最好的前几个答案
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:   # 达到我们要求的答案数目就停止
                break
            feature = features[pred.feature_index]  # 取出当前样本的feature

            if pred.start_index > 0:
                # 在处理后序列中截出的文本   # 预测的
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]

                # 在原始序列中截出的文本   # 预测的
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)  # 将预测id序列转为文本

                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())   # 用空格连着
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)

                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))

        if version_2_with_negative:
            # 考虑不可回答
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="no answer", start_logit=null_start_logit, end_logit=null_end_logit))

            # 在非常罕见的边缘情况下，我们只能有一个空预测。所以在这种情况下，我们只是创建一个nonce预测来避免失败。
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="no answer", start_logit=0.0, end_logit=0.0))

        if not nbest:
            nbest.append(_NbestPrediction(text="no answer", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1
        total_scores = []
        best_non_null_entry = None   # 保存最好的预测答案
        for entry in nbest:   # 前几个好的答案已经找到
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)   # 对所有预测的总分进行softmax

        nbest_json = []
        for i, entry in enumerate(nbest):
            output = collections.OrderedDict()
            output['text'] = entry.text
            output['probability'] = probs[i]
            output['start_logit'] = entry.start_logit
            output['end_logit'] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]['text']
        else:
            score_diff = score_null - best_non_null_entry.start_logit - best_non_null_entry.end_logit
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = "no answer"
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")

    with open(output_nbest_file, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(all_nbest_json, ensure_ascii=False, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(scores_diff_json, ensure_ascii=False, indent=4) + "\n")
    return all_predictions


# 计算得分
def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    # 1. 先处理标注的答案
    # 得到每个样本的真实答案
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    # 有答案的问题
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    # 无答案的问题
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}   # 给每个预测 加一个无答案的概率

    # 计算em、f1
    exact, f1 = get_raw_scores(examples, preds)

    # 计算em、f1阈值
    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

    return evaluation


def get_raw_scores(examples, preds):
    """
    计算em, f1
    """
    exact_scores = {}
    f1_scores = {}
    for example in examples:
        qas_id = example.qas_id

        # 标准答案  可能会标注多个答案
        gold_answers = [answer['text'] for answer in example.answers if normalize_answer(answer['text'])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = ['no answer']

        if qas_id not in preds:
            print('问题{}没有进行预测'.format(qas_id))
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)  # 预测答案和所有标注答案计算em  选最大
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)   # 预测答案和所有标注答案计算f1 选最大
    return exact_scores, f1_scores


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if s == '':
        return s
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# 以上的代码就把f1和em值计算完毕啦。


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]


def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh
    main_eval["has_ans_exact"] = has_ans_exact
    main_eval["has_ans_f1"] = has_ans_f1


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh


def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    has_ans_score, has_ans_cnt = 0, 0
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1

        if qid not in scores:
            continue
        has_ans_score += scores[qid]

    return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


# ###############################百度的评测################################
def baidu_evaluate(ref_file, pred_file, verbose=False):
    ref_ans = json.load(io.open(ref_file))
    pred_ans = json.load(io.open(pred_file))
    F1, EM, TOTAL, SKIP = evaluate(ref_ans, pred_ans, verbose=verbose)
    res = collections.OrderedDict()
    res['F1'] = round(F1, 4)
    res['EM'] = round(EM, 4)
    res['TOTAL'] = TOTAL
    res['SKIP'] = SKIP
    return res


def evaluate(ref_ans, pred_ans, verbose=False):
    """
    ref_ans: reference answers, dict
    pred_ans: predicted answer, dict
    return:
        f1_score: averaged F1 score
        em_score: averaged EM score
        total_count: number of samples in the reference dataset
        skip_count: number of samples skipped in the calculation due to unknown errors
    """
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    datas = ref_ans['data'][0]["paragraphs"]
    for document in datas:
        para = document['context'].strip()
        for qa in (document['qas']):
            total_count += 1
            query_id = qa['id']
            query_text = qa['question'].strip()
            answers = [a['text'] if a['text'] != '' else 'no answer' for a in qa['answers']]
            try:
                prediction = pred_ans[str(query_id)]
            except:
                skip_count += 1
                if verbose:
                    print("para: {}".format(para))
                    print("query: {}".format(query_text))
                    print("ref: {}".format('#'.join(answers)))
                    print("Skipped")
                    print('----------------------------')
                continue
            _f1 = calc_f1_score(answers, prediction)
            f1 += _f1
            em += calc_em_score(answers, prediction)
            if verbose:
                print("para: {}".format(para))
                print("query: {}".format(query_text))
                print("ref: {}".format('#'.join(answers)))
                print("cand: {}".format(prediction))
                print("score: {}".format(_f1))
                print('----------------------------')

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, prediction, debug=False):
    f1_scores = []
    for ans in answers:
        ans_segs = _tokenize_chinese_chars(_normalize(ans))
        prediction_segs = _tokenize_chinese_chars(_normalize(prediction))
        if debug:
            print(json.dumps(ans_segs, ensure_ascii=False))
            print(json.dumps(prediction_segs, ensure_ascii=False))
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        prec = 1.0*lcs_len/len(prediction_segs)
        rec = 1.0*lcs_len/len(ans_segs)
        f1 = (2 * prec * rec) / (prec + rec)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = _normalize(ans)
        prediction_ = _normalize(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def _tokenize_chinese_chars(text):
    """
    :param text: input text, unicode string
    :return:
        tokenized text, list
    """

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or char == "=":
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


def _normalize(in_str):
    """
    normalize the input unicode string
    """
    in_str = in_str.lower()
    sp_char = [
        u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
        u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
        u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def find_lcs(s1, s2):
    """find the longest common subsequence between s1 ans s2"""
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    max_len = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > max_len:
                    max_len = m[i+1][j+1]
                    p = i+1
    return s1[p-max_len:p], max_len

