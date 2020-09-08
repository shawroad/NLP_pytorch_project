# -*- coding: utf-8 -*-
# @Time    : 2020/7/8 17:26
# @Author  : xiaolu
# @FileName: data_process.py
# @Software: PyCharm

import random
from tqdm import tqdm
import spacy
import json
from collections import Counter
import numpy as np
import os.path
import os
import bisect
import torch
import re
from config import Config

nlp = spacy.blank("en")


def find_nearest(a, target, test_func=lambda x: True):
    idx = bisect.bisect_left(a, target)
    if (0 <= idx < len(a)) and a[idx] == target:
        return target, 0
    elif idx == 0:
        return a[0], abs(a[0] - target)
    elif idx == len(a):
        return a[-1], abs(a[-1] - target)
    else:
        d1 = abs(a[idx] - target) if test_func(a[idx]) else 1e200
        d2 = abs(a[idx-1] - target) if test_func(a[idx-1]) else 1e200
        if d1 > d2:
            return a[idx-1], d2
        else:
            return a[idx], d1


def fix_span(para, offsets, span):
    '''
    :param para: text_context
    :param offsets: offsets
    :param span: article['answer']
    :return:
    '''
    span = span.strip()
    parastr = "".join(para)
    assert span in parastr, '{}\t{}'.format(span, parastr)

    begins, ends = map(list, zip(*[y for x in offsets for y in x]))

    best_dist = 1e200
    best_indices = None

    if span == parastr:
        return parastr, (0, len(parastr)), 0

    for m in re.finditer(re.escape(span), parastr):
        begin_offset, end_offset = m.span()

        fixed_begin, d1 = find_nearest(begins, begin_offset, lambda x: x < end_offset)
        fixed_end, d2 = find_nearest(ends, end_offset, lambda x: x > begin_offset)

        if d1 + d2 < best_dist:
            best_dist = d1 + d2
            best_indices = (fixed_begin, fixed_end)
            if best_dist == 0:
                break
    assert best_indices is not None
    return parastr[best_indices[0]:best_indices[1]], best_indices, best_dist


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        pre = current
        current = text.find(token, current)
        if current < 0:
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None, token2idx_dict=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]  # 过滤低频词

    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        # 这一种是没有加载glove词向量  自己随机初始化词向量
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 2)} if token2idx_dict is None else token2idx_dict

    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]

    idx2token_dict = {idx: token for token, idx in token2idx_dict.items()}

    return emb_mat, token2idx_dict, idx2token_dict


def build_features(examples, data_type, out_file, word2idx_dict, char2idx_dict):
    '''
    examples, data_split, record_file, word2idx_dict, char2idx_dict
    :param examples: 样本
    :param data_type: 当前处理的类型  训练集, 测试集, 验证集
    :param out_file: 处理完输出的文件夹
    :param word2idx_dict: 词表
    :param char2idx_dict: 字表
    :return:
    '''
    if data_type == 'test':
        # 如果是测试集  统计一下段落和问题的最大长度
        para_limit, ques_limit = 0, 0
        for example in tqdm(examples):
            para_limit = max(para_limit, len(example['context_tokens']))
            ques_limit = max(ques_limit, len(example['ques_tokens']))
    else:
        para_limit = Config.para_limit
        ques_limit = Config.ques_limit

    char_limit = Config.char_limit

    # 将不符合我们长度约束的语料过滤掉
    def filter_func(example):
        return len(example['context_tokens']) > para_limit or len(example['ques_tokens']) > ques_limit

    print("Processing {} examples...".format(data_type))
    datapoints = []
    total = 0   # 统计的是真是有效语料的条数
    total_ = 0   # 统计的所有语料的条数
    for example in tqdm(examples):
        total_ += 1
        if filter_func(example):
            continue
        total += 1

        context_idxs = np.zeros(para_limit, dtype=np.int64)
        context_char_idxs = np.zeros((para_limit, char_limit), dtype=np.int64)

        ques_idxs = np.zeros(ques_limit, dtype=np.int64)
        ques_char_idxs = np.zeros((ques_limit, char_limit), dtype=np.int64)

        def _get_word(word):
            # 将每个词转为id
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            # 将每个字符转为id
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        # 对于词  文章和问题转为id
        context_idxs[:len(example['context_tokens'])] = [_get_word(token) for token in example['context_tokens']]
        ques_idxs[:len(example['ques_tokens'])] = [_get_word(token) for token in example['ques_tokens']]

        # 对于字  文章和问题转为id
        for i, token in enumerate(example["context_chars"]):
            l = min(len(token), char_limit)
            context_char_idxs[i, :l] = [_get_char(char) for char in token[:l]]

        for i, token in enumerate(example["ques_chars"]):
            l = min(len(token), char_limit)
            ques_char_idxs[i, :l] = [_get_char(char) for char in token[:l]]

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1, y2 = start, end

        datapoints.append({
            'context_idxs': torch.from_numpy(context_idxs),
            'context_char_idxs': torch.from_numpy(context_char_idxs),
            'ques_idxs': torch.from_numpy(ques_idxs),
            'ques_char_idxs': torch.from_numpy(ques_char_idxs),
            'y1': y1,
            'y2': y2,
            'id': example['id'],
            'start_end_facts': example['start_end_facts']})
    print("Build {} / {} instances of features in total".format(total, total_))
    torch.save(datapoints, out_file)


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    with open(filename, "w", encoding='utf8') as fh:
        json.dump(obj, fh)


def _process_article(article):
    # 1. 取出文章
    paragraphs = article['context']

    if len(paragraphs) == 0:
        paragraphs = [['some random title', 'some random stuff']]

    text_context, context_tokens, context_chars = '', [], []
    offsets = []
    flat_offsets = []
    start_end_facts = [] # (start_token_id, end_token_id, is_sup_fact=True/False)
    sent2title_ids = []

    def _process(sent, is_sup_fact, is_title=False):
        '''
        :param sent:
        :param is_sup_fact:
        :param is_title:
        :return:
        '''
        nonlocal text_context, context_tokens, context_chars, offsets, start_end_facts, flat_offsets
        N_chars = len(text_context)

        sent = sent
        sent_tokens = word_tokenize(sent)
        if is_title:
            sent = '<t> {} </t>'.format(sent)
            sent_tokens = ['<t>'] + sent_tokens + ['</t>']
        sent_chars = [list(token) for token in sent_tokens]
        sent_spans = convert_idx(sent, sent_tokens)

        sent_spans = [[N_chars+e[0], N_chars+e[1]] for e in sent_spans]
        N_tokens, my_N_tokens = len(context_tokens), len(sent_tokens)
        text_context += sent
        context_tokens.extend(sent_tokens)
        context_chars.extend(sent_chars)
        start_end_facts.append((N_tokens, N_tokens+my_N_tokens, is_sup_fact))
        offsets.append(sent_spans)
        flat_offsets.extend(sent_spans)

    if 'supporting_facts' in article:
        sp_set = set(list(map(tuple, article['supporting_facts'])))  # article['supporting_facts'] (相关标题, 相关句子的编号)
        # print(sp_set)    #  {('First for Women', 0), ("Arthur's Magazine", 0)}
    else:
        sp_set = set()

    sp_fact_cnt = 0
    for para in paragraphs:   # 遍历文章中的几段话
        cur_title, cur_para = para[0], para[1]
        _process(cur_title, False, is_title=True)  # 处理标题
        # print(context_tokens)   # ['<t>', 'Radio', 'City', '(', 'Indian', 'radio', 'station', ')', '</t>']
        # print(context_chars)   # [['<', 't', '>'], ['R', 'a', 'd', 'i', 'o'], ['C', 'i', 't', 'y'], ...]
        # print(start_end_facts)    # [(0, 9, False)]
        # print(offsets)
        # print(flat_offsets)   # 每个单词的起始和结束[[0, 3], [4, 9], [10, 14], [15, 16], [16, 22]]

        sent2title_ids.append((cur_title, -1))
        for sent_id, sent in enumerate(cur_para):  # 处理每段话
            is_sup_fact = (cur_title, sent_id) in sp_set  # 看当段的当句话是不是相关句
            if is_sup_fact:
                sp_fact_cnt += 1   # 统计相关句子个数
            _process(sent, is_sup_fact)
            sent2title_ids.append((cur_title, sent_id))  # 标记的是每句话是所在的哪个标题下的第几话

    # 处理答案
    if 'answer' in article:
        answer = article['answer'].strip()
        if answer.lower() == 'yes':
            best_indices = [-1, -1]   # 单独考虑yes or no
        elif answer.lower() == 'no':
            best_indices = [-2, -2]
        else:
            if article['answer'].strip() not in ''.join(text_context):
                best_indices = (0, 1)   # 答案不存在
            else:
                _, best_indices, _ = fix_span(text_context, offsets, article['answer'])
                answer_span = []
                for idx, span in enumerate(flat_offsets):
                    if not (best_indices[1] <= span[0] or best_indices[0] >= span[1]):
                        answer_span.append(idx)
                best_indices = (answer_span[0], answer_span[-1])
    else:
        answer = 'random'
        best_indices = (0, 1)

    # 处理问题
    ques_tokens = word_tokenize(article['question'])
    ques_chars = [list(token) for token in ques_tokens]

    example = {'context_tokens': context_tokens,
               'context_chars': context_chars,
               'ques_tokens': ques_tokens,
               'ques_chars': ques_chars,
               'y1s': [best_indices[0]],
               'y2s': [best_indices[1]],
               'id': article['_id'],
               'start_end_facts': start_end_facts}   # [[(0, 9, False), (9, 27, False), ...(902, 923, False)]

    eval_example = {'context': text_context,   # 所有段落拼接起来的文章
                    'spans': flat_offsets,   #
                    'answer': [answer],    # 答案文本
                    'id': article['_id'],
                    'sent2title_ids': sent2title_ids}   # 每句话在那个标题下，并且在当前标题下的第几句
    return example, eval_example


def process_file(data_path, word_counter=None, char_counter=None):
    data = json.load(open(data_path, 'r', encoding='utf8'))

    eval_examples = {}
    # 处理文章
    print('正在分解文章.... 包括(分词, 分字符, 处理每句话在文章的起始和结束位置...)')
    outputs = []
    for article in tqdm(data):
        temp = _process_article(article)
        outputs.append(temp)
    examples = [e[0] for e in outputs]

    for _, e in outputs:
        if e is not None:
            eval_examples[e['id']] = e   # id -> 一条数据

    if word_counter is not None and char_counter is not None:
        for example in examples:
            for token in example['ques_tokens'] + example['context_tokens']:
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1
    random.shuffle(examples)
    print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def prepro(data_split):
    if data_split == 'train':
        word_counter, char_counter = Counter(), Counter()
        examples, eval_examples = process_file(Config.train_data_path, word_counter, word_counter)
    else:
        examples, eval_examples = process_file(Config.test_data_path)

    # 构造词典 并加载词向量
    word2idx_dict = None
    if os.path.isfile(Config.word2idx_file):
        with open(Config.word2idx_file, 'r', encoding='utf8') as f:
            word2idx_dict = json.load(f)
    else:
        word_emb_mat, word2idx_dict, idx2word_dict = get_embedding(word_counter, 'word',
                                                                   emb_file=Config.glove_word_file,
                                                                   size=Config.glove_word_size,
                                                                   vec_size=Config.glove_dim,
                                                                   token2idx_dict=word2idx_dict)
    # 构造字典
    char2idx_dict = None
    if os.path.isfile(Config.char2idx_file):
        with open(Config.char2idx_file, 'r', encoding='utf8') as f:
            char2idx_dict = json.load(f)
    else:
        char_emb_mat, char2idx_dict, idx2char_dict = get_embedding(char_counter, 'char',
                                                                   emb_file=None, size=None,
                                                                   vec_size=Config.char_dim,
                                                                   token2idx_dict=char2idx_dict)
    if data_split == 'train':
        record_file = Config.train_record_file
        eval_file = Config.train_eval_file
    elif data_split == 'dev':
        record_file = Config.dev_record_file
        eval_file = Config.dev_eval_file
    elif data_split == 'test':
        record_file = Config.test_record_file
        eval_file = Config.test_eval_file

    # 将文本转为token
    build_features(examples, data_split, record_file, word2idx_dict, char2idx_dict)
    save(eval_file, eval_examples, message='{} eval'.format(data_split))

    if not os.path.isfile(Config.word2idx_file):
        save(Config.word_emb_file, word_emb_mat, message="word embedding")
        save(Config.char_emb_file, char_emb_mat, message="char embedding")
        save(Config.word2idx_file, word2idx_dict, message="word2idx")
        save(Config.char2idx_file, char2idx_dict, message="char2idx")
        save(Config.idx2word_file, idx2word_dict, message='idx2word')
        save(Config.idx2char_file, idx2char_dict, message='idx2char')


if __name__ == '__main__':
    # 预处理训练集
    data_split = 'train'
    prepro(data_split)

    # 预处理验证集
    data_split = 'dev'
    prepro(data_split)
