# -*- coding: utf-8 -*-
# @Time    : 2020/7/20 14:30
# @Author  : xiaolu
# @FileName: data_process1.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2020/6/30 11:03
# @Author  : xiaolu
# @FileName: step_1_data_process.py
# @Software: PyCharm

import argparse
import json
import gzip
import pickle
from tqdm import tqdm
from transformers import BertTokenizer


class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 doc_tokens,
                 question_text,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 para_start_end_position,
                 sent_start_end_position,
                 entity_start_end_position,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.entity_start_end_position = entity_start_end_position
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 sent_spans,
                 sup_fact_ids,
                 ans_type,
                 token_to_orig_map,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

        self.sent_spans = sent_spans
        self.sup_fact_ids = sup_fact_ids
        self.ans_type = ans_type
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position


def check_in_full_paras(answer, paras):
    full_doc = ""
    for p in paras:
        full_doc += " ".join(p[1])
    return answer in full_doc


def read_examples(full_file):
    # 加载数据
    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    cnt = 0
    examples = []
    xl = 0
    for case in tqdm(full_data):
        #
        key = case['_id']     # 文章id
        qas_type = ""    # case['type']
        sup_facts = set([(sp[0], sp[1]) for sp in case['supporting_facts']])   # 段落标题  相关句的编号
        sup_titles = set([sp[0] for sp in case['supporting_facts']])   # 当前这条语料中的标题进行去重
        orig_answer_text = case['answer']    # 获取原始答案的文本

        sent_id = 0
        doc_tokens = []
        sent_names = []
        sup_facts_sent_id = []
        sent_start_end_position = []
        para_start_end_position = []
        entity_start_end_position = []   # 没啥乱用  对接的是DFGN中的实体
        ans_start_position, ans_end_position = [], []

        JUDGE_FLAG = orig_answer_text == 'yes' or orig_answer_text == 'no' or orig_answer_text == 'unknown' or orig_answer_text == ""  # judge_flag??
        FIND_FLAG = False   # 标志是否找到答案了

        char_to_word_offset = []  # Accumulated along all sentences
        prev_is_whitespace = True

        # for debug
        titles = set()
        para_data = case['context']   # 这个para_data  可能包含多段文本 以及多个标题
        for paragraph in para_data:   # 遍历段落
            title = paragraph[0]    # 标题
            sents = paragraph[1]    # 段落的每句话

            titles.add(title)   # 将当前段落的标题加入到标题集合中
            is_gold_para = 1 if title in sup_titles else 0   # is_gold_para=1说明答案与当前的段落有关系 否则没关系

            para_start_position = len(doc_tokens)   # 段落的起始位置

            for local_sent_id, sent in enumerate(sents):   # 遍历段落中的句子
                if local_sent_id >= 100:   # 句子的个数超过100条就不要了
                    break
                # Determine the global sent id for supporting facts
                local_sent_name = (title, local_sent_id)   # (标题, 句子id)
                sent_names.append(local_sent_name)
                if local_sent_name in sup_facts:
                    sup_facts_sent_id.append(sent_id)   # 相关的句子id
                sent_id += 1
                sent = " ".join(sent)   # 句子中的字之间加入空格
                sent += " "

                sent_start_word_id = len(doc_tokens)
                sent_start_char_id = len(char_to_word_offset)

                for c in sent:   # 遍历句子中的每个字
                    if is_whitespace(c):
                        prev_is_whitespace = True    # 这里是标志空格，换行等不占位
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False

                    char_to_word_offset.append(len(doc_tokens) - 1)

                sent_end_word_id = len(doc_tokens) - 1  # 当前句子结束的标志
                sent_start_end_position.append((sent_start_word_id, sent_end_word_id))   # 句子的起始和结束位置

                # 标记答案
                answer_offsets = []
                offset = -1
                tmp_answer = " ".join(orig_answer_text)   # 将原始答案展开  即在字之间加入空格
                while True:
                    offset = sent.find(tmp_answer, offset + 1)   # 当前句子中找答案的位置  起始位置(基于当前句子)
                    if offset != -1:
                        answer_offsets.append(offset)
                    else:
                        break   # 没有找到答案

                # answer_offsets = [m.start() for m in re.finditer(orig_answer_text, sent)]
                if not JUDGE_FLAG and not FIND_FLAG and len(answer_offsets) > 0:
                    FIND_FLAG = True
                    for answer_offset in answer_offsets:
                        start_char_position = sent_start_char_id + answer_offset   # 起始位置  句子的起始位置+offset
                        end_char_position = start_char_position + len(tmp_answer) - 1
                        ans_start_position.append(char_to_word_offset[start_char_position])
                        ans_end_position.append(char_to_word_offset[end_char_position])

                if len(doc_tokens) > 430:
                    break
            para_end_position = len(doc_tokens) - 1

            para_start_end_position.append((para_start_position, para_end_position, title, is_gold_para))

        if len(ans_end_position) > 1:
            cnt += 1
        # if key < 10:
        #     print("qid {}".format(key))
        #     print("qas type {}".format(qas_type))
        #     print("doc tokens {}".format(doc_tokens))   # 中文的token列表
        #     print("question {}".format(case['question']))   # 问题
        #     print("sent num {}".format(sent_id + 1))   # 句子的个数
        #     print("sup face id {}".format(sup_facts_sent_id))   # 相关句子id
        #     print("para_start_end_position {}".format(para_start_end_position))   # 段落起始和结束位置
        #     print("sent_start_end_position {}".format(sent_start_end_position))   # 句子的起始和结束位置
        #     print("entity_start_end_position {}".format(entity_start_end_position))   # 没啥用的东西
        #     print("orig_answer_text {}".format(orig_answer_text))    # 原始答案文本
        #     print("ans_start_position {}".format(ans_start_position))   # 起始位置
        #     print("ans_end_position {}".format(ans_end_position))   # 结束位置
        '''
        qid 0
        qas type
        doc tokens ['经', '审', '理', '查', '明', '：', '被', '告', '铁', 'x', '3', '系', '北', '京', '市', '东', '城', '区', '街', '号', '楼', '单', '元', '号', '房', '屋', '业', '主', '，', '由', '原', '告', '为', '该', '房', '屋', '提', '供', '冬', '季', '供', '暖', '服', '务', '，', '供', '暖', '费', '缴', '纳', '标', '准', '为', '每', '年', '2', '2', '5', '7', '.', '5', '元', '，', '双', '方', '未', '提', '供', '书', '面', '的', '供', '暖', '合', '同', '。', '2', '0', '0', '5', '年', '1', '1', '月', '至', '2', '0', '1', '3', '年', '3', '月', '期', '间', '的', '供', '暖', '费', '被', '告', '至', '今', '未', '向', '原', '告', '交', '纳', '。', '被', '告', '提', '供', '了', '其', '房', '屋', '在', '2', '0', '0', '6', '年', '1', '2', '月', '7', '日', '的', '报', '修', '单', '，', '保', '修', '单', '中', '测', '温', '结', '果', '显', '示', '近', '端', '1', '5', '.', '2', '，', '中', '端', '1', '4', '.', '9', '，', '末', '端', '1', '2', '.', '8', '，', '原', '告', '对', '该', '测', '温', '记', '录', '予', '以', '认', '可', '，', '但', '原', '告', '认', '为', '被', '告', '房', '屋', '其', '他', '时', '间', '的', '温', '度', '均', '已', '达', '标', '；', '被', '告', '主', '张', '其', '房', '屋', '历', '年', '供', '暖', '温', '度', '均', '是', '如', '此', '。', '现', '原', '告', '诉', '至', '法', '院', '，', '诉', '如', '所', '请', '，', '被', '告', '坚', '持', '其', '答', '辩', '意', '见', '，', '经', '调', '解', '，', '双', '方', '各', '执', '己', '见', '。', '上', '述', '事', '实', '，', '有', '双', '方', '的', '陈', '述', '，', '北', '京', '市', '供', '热', '运', '行', '单', '位', '备', '案', '登', '记', '证', '，', '上', '水', '记', '录', '，', '供', '暖', '费', '催', '缴', '单', '，', '报', '修', '单', '等', '在', '案', '佐', '证', '。']
        question 保修单中测温结果的末端温度？
        sent num 24
        sup face id [6, 8]
        para_start_end_position [(0, 297, '经审理查明：被告铁x3系北京市东城区街号楼单元号房屋业主，', 1)]
        sent_start_end_position [(0, 28), (29, 44), (45, 62), (63, 75), (76, 108), (109, 132), (133, 149), (150, 156), (157, 163), (164, 176), (177, 197), (198, 215), (216, 223), (224, 228), (229, 238), (239, 242), (243, 249), (250, 254), (255, 261), (262, 276), (277, 281), (282, 288), (289, 297)]
        entity_start_end_position []
        orig_answer_text 12.8
        ans_start_position [159]
        ans_end_position [162]
        '''
        example = Example(
            qas_id=key,
            qas_type=qas_type,
            doc_tokens=doc_tokens,
            question_text=case['question'],
            sent_num=sent_id + 1,
            sent_names=sent_names,
            sup_fact_id=sup_facts_sent_id,
            para_start_end_position=para_start_end_position,
            sent_start_end_position=sent_start_end_position,
            entity_start_end_position=entity_start_end_position,
            orig_answer_text=orig_answer_text,
            start_position=ans_start_position,
            end_position=ans_end_position)
        examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    '''
    将语料转为id序列
    :param examples:
    :param tokenizer:
    :param max_seq_length:
    :param max_query_length:
    :return:
    '''
    # max_query_length = 50
    features = []
    failed = 0
    for (example_index, example) in enumerate(tqdm(examples)):
        # 加个四分类  主要处理yes or no or unknown or exist_answer
        if example.orig_answer_text == 'yes':
            ans_type = 1
        elif example.orig_answer_text == 'no':
            ans_type = 2
        elif example.orig_answer_text == 'unknown':
            ans_type = 3
        else:
            ans_type = 0  # 统计answer type

        # 前半部分的问题处理
        query_tokens = ["[CLS]"]
        for token in example.question_text.split(' '):
            query_tokens.extend(tokenizer.tokenize(token))
        if len(query_tokens) > max_query_length - 1:
            query_tokens = query_tokens[:max_query_length - 1]
        query_tokens.append("[SEP]")

        # para_spans = []
        # entity_spans = []
        sentence_spans = []
        all_doc_tokens = []
        orig_to_tok_index = []
        orig_to_tok_back_index = []
        tok_to_orig_index = [0] * len(query_tokens)

        # 问题+文章
        all_doc_tokens = ["[CLS]"]
        for token in example.question_text.split(' '):
            all_doc_tokens.extend(tokenizer.tokenize(token))
        if len(all_doc_tokens) > max_query_length - 1:
            all_doc_tokens = all_doc_tokens[:max_query_length - 1]
        all_doc_tokens.append("[SEP]")

        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))  # [0, 1, 2, 3, 4, 5....]
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
            orig_to_tok_back_index.append(len(all_doc_tokens) - 1)

        def relocate_tok_span(orig_start_position, orig_end_position, orig_text):
            '''
            :param orig_start_position: 原始答案的起始位置
            :param orig_end_position: 原始答案的结束位置
            :param orig_text: 原始的答案
            :return:
            '''

            if orig_start_position is None:
                return 0, 0

            tok_start_position = orig_to_tok_index[orig_start_position]
            if orig_end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[orig_end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            return _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, orig_text)

        ans_start_position, ans_end_position = [], []
        for ans_start_pos, ans_end_pos in zip(example.start_position, example.end_position):
            s_pos, e_pos = relocate_tok_span(ans_start_pos, ans_end_pos, example.orig_answer_text)
            ans_start_position.append(s_pos)
            ans_end_position.append(e_pos)

        for sent_span in example.sent_start_end_position:
            if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
                continue
            sent_start_position = orig_to_tok_index[sent_span[0]]
            sent_end_position = orig_to_tok_back_index[sent_span[1]]
            sentence_spans.append((sent_start_position, sent_end_position))

        all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + ["[SEP]"]
        doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)  # 转为id序列   问题+文章
        query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)   # 转为id序列   只有问题

        doc_input_mask = [1] * len(doc_input_ids)
        doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))

        while len(doc_input_ids) < max_seq_length:
            # padding
            doc_input_ids.append(0)
            doc_input_mask.append(0)
            doc_segment_ids.append(0)

        query_input_mask = [1] * len(query_input_ids)
        query_segment_ids = [0] * len(query_input_ids)

        while len(query_input_ids) < max_query_length:
            query_input_ids.append(0)
            query_input_mask.append(0)
            query_segment_ids.append(0)

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert len(query_input_ids) == max_query_length
        assert len(query_input_mask) == max_query_length
        assert len(query_segment_ids) == max_query_length

        sentence_spans = get_valid_spans(sentence_spans, max_seq_length)

        sup_fact_ids = example.sup_fact_id
        sent_num = len(sentence_spans)
        sup_fact_ids = [sent_id for sent_id in sup_fact_ids if sent_id < sent_num]
        if len(sup_fact_ids) != len(example.sup_fact_id):
            failed += 1
        if example.qas_id < 10:
            print("qid {}".format(example.qas_id))
            print("all_doc_tokens {}".format(all_doc_tokens))
            print("doc_input_ids {}".format(doc_input_ids))
            print("doc_input_mask {}".format(doc_input_mask))
            print("doc_segment_ids {}".format(doc_segment_ids))
            print("query_tokens {}".format(query_tokens))
            print("query_input_ids {}".format(query_input_ids))
            print("query_input_mask {}".format(query_input_mask))
            print("query_segment_ids {}".format(query_segment_ids))
            print("sentence_spans {}".format(sentence_spans))
            print("sup_fact_ids {}".format(sup_fact_ids))
            print("ans_type {}".format(ans_type))
            print("tok_to_orig_index {}".format(tok_to_orig_index))
            print("ans_start_position {}".format(ans_start_position))
            print("ans_end_position {}".format(ans_end_position))
        '''
        qid 0
        all_doc_tokens ['[CLS]', '保', '修', '单', '中', '测', '温', '结', '果', '的', '末', '端', '温', '度', '？', '[SEP]', '经', '审', '理', '查', '明', '：', '被', '告', '铁', 'x', '3', '系', '北', '京', '市', '东', '城', '区', '街', '号', '楼', '单', '元', '号', '房', '屋', '业', '主', '，', '由', '原', '告', '为', '该', '房', '屋', '提', '供', '冬', '季', '供', '暖', '服', '务', '，', '供', '暖', '费', '缴', '纳', '标', '准', '为', '每', '年', '2', '2', '5', '7', '.', '5', '元', '，', '双', '方', '未', '提', '供', '书', '面', '的', '供', '暖', '合', '同', '。', '2', '0', '0', '5', '年', '1', '1', '月', '至', '2', '0', '1', '3', '年', '3', '月', '期', '间', '的', '供', '暖', '费', '被', '告', '至', '今', '未', '向', '原', '告', '交', '纳', '。', '被', '告', '提', '供', '了', '其', '房', '屋', '在', '2', '0', '0', '6', '年', '1', '2', '月', '7', '日', '的', '报', '修', '单', '，', '保', '修', '单', '中', '测', '温', '结', '果', '显', '示', '近', '端', '1', '5', '.', '2', '，', '中', '端', '1', '4', '.', '9', '，', '末', '端', '1', '2', '.', '8', '，', '原', '告', '对', '该', '测', '温', '记', '录', '予', '以', '认', '可', '，', '但', '原', '告', '认', '为', '被', '告', '房', '屋', '其', '他', '时', '间', '的', '温', '度', '均', '已', '达', '标', '；', '被', '告', '主', '张', '其', '房', '屋', '历', '年', '供', '暖', '温', '度', '均', '是', '如', '此', '。', '现', '原', '告', '诉', '至', '法', '院', '，', '诉', '如', '所', '请', '，', '被', '告', '坚', '持', '其', '答', '辩', '意', '见', '，', '经', '调', '解', '，', '双', '方', '各', '执', '己', '见', '。', '上', '述', '事', '实', '，', '有', '双', '方', '的', '陈', '述', '，', '北', '京', '市', '供', '热', '运', '行', '单', '位', '备', '案', '登', '记', '证', '，', '上', '水', '记', '录', '，', '供', '暖', '费', '催', '缴', '单', '，', '报', '修', '单', '等', '在', '案', '佐', '证', '。', '[SEP]']
        doc_input_ids [101, 924, 934, 1296, 704, 3844, 3946, 5310, 3362, 4638, 3314, 4999, 3946, 2428, 8043, 102, 5307, 2144, 4415, 3389, 3209, 8038, 6158, 1440, 7188, 166, 124, 5143, 1266, 776, 2356, 691, 1814, 1277, 6125, 1384, 3517, 1296, 1039, 1384, 2791, 2238, 689, 712, 8024, 4507, 1333, 1440, 711, 6421, 2791, 2238, 2990, 897, 1100, 2108, 897, 3265, 3302, 1218, 8024, 897, 3265, 6589, 5373, 5287, 3403, 1114, 711, 3680, 2399, 123, 123, 126, 128, 119, 126, 1039, 8024, 1352, 3175, 3313, 2990, 897, 741, 7481, 4638, 897, 3265, 1394, 1398, 511, 123, 121, 121, 126, 2399, 122, 122, 3299, 5635, 123, 121, 122, 124, 2399, 124, 3299, 3309, 7313, 4638, 897, 3265, 6589, 6158, 1440, 5635, 791, 3313, 1403, 1333, 1440, 769, 5287, 511, 6158, 1440, 2990, 897, 749, 1071, 2791, 2238, 1762, 123, 121, 121, 127, 2399, 122, 123, 3299, 128, 3189, 4638, 2845, 934, 1296, 8024, 924, 934, 1296, 704, 3844, 3946, 5310, 3362, 3227, 4850, 6818, 4999, 122, 126, 119, 123, 8024, 704, 4999, 122, 125, 119, 130, 8024, 3314, 4999, 122, 123, 119, 129, 8024, 1333, 1440, 2190, 6421, 3844, 3946, 6381, 2497, 750, 809, 6371, 1377, 8024, 852, 1333, 1440, 6371, 711, 6158, 1440, 2791, 2238, 1071, 800, 3198, 7313, 4638, 3946, 2428, 1772, 2347, 6809, 3403, 8039, 6158, 1440, 712, 2476, 1071, 2791, 2238, 1325, 2399, 897, 3265, 3946, 2428, 1772, 3221, 1963, 3634, 511, 4385, 1333, 1440, 6401, 5635, 3791, 7368, 8024, 6401, 1963, 2792, 6435, 8024, 6158, 1440, 1780, 2898, 1071, 5031, 6796, 2692, 6224, 8024, 5307, 6444, 6237, 8024, 1352, 3175, 1392, 2809, 2346, 6224, 511, 677, 6835, 752, 2141, 8024, 3300, 1352, 3175, 4638, 7357, 6835, 8024, 1266, 776, 2356, 897, 4178, 6817, 6121, 1296, 855, 1906, 3428, 4633, 6381, 6395, 8024, 677, 3717, 6381, 2497, 8024, 897, 3265, 6589, 998, 5373, 1296, 8024, 2845, 934, 1296, 5023, 1762, 3428, 858, 6395, 511, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        doc_input_mask [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        doc_segment_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        query_tokens ['[CLS]', '保', '修', '单', '中', '测', '温', '结', '果', '的', '末', '端', '温', '度', '？', '[SEP]']
        query_input_ids [101, 2314, 3844, 3946, 5310, 3362, 4638, 3314, 4999, 3946, 2428, 8043, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        query_input_mask [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        query_segment_ids [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sentence_spans [(16, 44), (45, 60), (61, 78), (79, 91), (92, 124), (125, 148), (149, 165), (166, 172), (173, 179), (180, 192), (193, 213), (214, 231), (232, 239), (240, 244), (245, 254), (255, 258), (259, 265), (266, 270), (271, 277), (278, 292), (293, 297), (298, 304), (305, 313)]
        sup_fact_ids [6, 8]
        ans_type 0
        tok_to_orig_index [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297]
        ans_start_position [175]
        ans_end_position [178]
        '''
        features.append(
            InputFeatures(qas_id=example.qas_id,
                          doc_tokens=all_doc_tokens,
                          doc_input_ids=doc_input_ids,
                          doc_input_mask=doc_input_mask,
                          doc_segment_ids=doc_segment_ids,
                          query_tokens=query_tokens,
                          query_input_ids=query_input_ids,
                          query_input_mask=query_input_mask,
                          query_segment_ids=query_segment_ids,
                          sent_spans=sentence_spans,
                          sup_fact_ids=sup_fact_ids,
                          ans_type=ans_type,
                          token_to_orig_map=tok_to_orig_index,
                          start_position=ans_start_position,
                          end_position=ans_end_position)
        )
    return features


def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx


def get_valid_spans(spans, limit):
    new_spans = []
    for span in spans:
        if span[1] < limit:
            new_spans.append(span)
        else:
            new_span = list(span)
            new_span[1] = limit - 1
            new_spans.append(tuple(new_span))
            break
    return new_spans


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 两种输出
    parser.add_argument("--example_output", required=True, type=str)
    parser.add_argument("--feature_output", required=True, type=str)

    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--full_data", type=str, required=True)
    parser.add_argument('--tokenizer_path', default='./albert_pretrain/vocab.txt', type=str)

    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

    examples = read_examples(full_file=args.full_data)
    with gzip.open(args.example_output, 'wb') as fout:
        pickle.dump(examples, fout)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length=512, max_query_length=50)
    with gzip.open(args.feature_output, 'wb') as fout:
        pickle.dump(features, fout)
