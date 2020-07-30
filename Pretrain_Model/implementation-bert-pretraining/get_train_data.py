# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 10:10
# @Author  : xiaolu
# @FileName: get_train_data.py
# @Software: PyCharm
from transformers import BertTokenizer
import random
import numpy as np
import collections
import json

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def create_masked_lm_predictions(tokens, max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""

    # n-gram masking Albert
    ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_ngram + 1)
    pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        cand_indices.append(i)
    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    random.shuffle(cand_indices)
    masked_token_labels = []
    covered_indices = set()
    for index in cand_indices:
        n = np.random.choice(ngrams, p=pvals)
        if len(masked_token_labels) >= num_to_mask:
            break
        if index in covered_indices:
            continue
        if index < len(cand_indices) - (n - 1):
            for i in range(n):
                ind = index + i
                if ind in covered_indices:
                    continue
                covered_indices.add(ind)
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if random.random() < 0.5:
                        masked_token = tokens[ind]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = random.choice(vocab_list)
                masked_token_labels.append(MaskedLmInstance(index=ind, label=tokens[ind]))
                tokens[ind] = masked_token
    # assert len(masked_token_labels) <= num_to_mask
    masked_token_labels = sorted(masked_token_labels, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_token_labels]
    masked_labels = [p.label for p in masked_token_labels]
    return tokens, mask_indices, masked_labels


def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob,
                                   max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_words):
    # 将单文档处理成训练数据
    document = all_documents[document_index]  # 得到一个文档

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3   # max_seq_length 最大序列长度 最长设置到512, 可根据训练数据集进行调整

    target_seq_length = max_num_tokens

    # 有一定的比例，如10%的概率，我们使用比较短的序列长度，以缓解预训练的长序列和调优阶段（可能的）短序列的不一致情况
    if random.random() < short_seq_prob:    # 有10%选用短文本
        target_seq_length = random.randint(2, max_num_tokens)

    # 设法使用实际的句子，而不是任意的截断句子，从而更好的构造句子连贯性预测的任务
    instances = []
    current_chunk = []  # 当前处理的文本段，包含多个句子
    current_length = 0
    i = 0
    # print("###document:",document) # 一个document可以是一整篇文章、新闻、词条等.
    while i < len(document):  # 从文档的开始看起  一句话一句话的看
        segment = document[i]   # segment是列表，代表的是按字分开的一个完整句子

        # segment = get_new_segment(segment)  # whole word mask for chinese: 结合分词的中文的whole mask设置即在需要的地方加上“##”
        current_chunk.append(segment)  # 将一个独立的句子加入到当前的文本块中
        current_length += len(segment)  # 累计到为止位置接触到句子的总长度
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:  # 如果当前块不空
                a_end = 1
                if len(current_chunk) >= 2:   # 当前块，如果包含超过两个句子，取当前块的一部分作为“A[SEP]B“中的A部分
                    a_end = random.randint(1, len(current_chunk) - 1)
                # 将当前文本段中选取出来的前半部分，赋值给A即tokens_a
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                # 构造“A[SEP]B“中的B部分(有一部分是正常的当前文档中的后半部;在原BERT的实现中一部分是随机的从另一个文档中选取的，）
                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])

                # 有百分之50%的概率交换一下tokens_a和tokens_b的位置
                # print("tokens_a length1:",len(tokens_a))
                # print("tokens_b length1:",len(tokens_b)) # len(tokens_b) = 0
                if len(tokens_a) == 0 or len(tokens_b) == 0:
                    continue
                if random.random() < 0.5:  # 交换一下tokens_a和tokens_b
                    is_random_next = True
                    temp = tokens_a
                    tokens_a = tokens_b
                    tokens_b = temp
                else:
                    is_random_next = False

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                # 把tokens_a & tokens_b加入到按照bert的风格，即以[CLS]tokens_a[SEP]tokens_b[SEP]的形式，结合到一起，作为最终的tokens;
                # 也带上segment_ids，前面部分segment_ids的值是0，后面部分的值是1.
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                # 创建masked LM的任务的数据
                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_words)
                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels
                }
                '''
                {'tokens': ['[CLS]', '肥', '市', '庐', '[MASK]', '[MASK]', '中', '铁', '国', '际', '城', '湿', '地', '公', '园', '散', '步', '时', '[MASK]', '[MASK]', '[MASK]', '湿', '地', '公', '园', '临', '近', '出', '口', '（', '出', '口', '位', '[MASK]', '东', '西', '走', '向', '的', '清', '源', '路', '）', '的', '内', '部', '道', '路', '时', '，', '被', '中', 'x5', '物', '业', '管', '理', '公', '司', '合', '肥', '##棗', '公', '司', '[SEP]', '人', '陈', '述', '和', '002', '##座', '[MASK]', '确', '认', '的', '证', '据', '[MASK]', '本', '院', '认', '定', '事', '实', '如', '下', '：', '韩', '家', '麒', '[MASK]', '公', '民', '身', '[MASK]', '号', '码', '##ute', '户', '[MASK]', '登', '记', '类', '别', '为', '非', '农', '业', '家', '庭', '户', '）', '与', '孔', 'x', '##0', '系', '夫', '妻', '关', '系', '[MASK]', '[MASK]', '[MASK]', '生', '育', '有', '[SEP]'], 
                 'segment_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                 'is_random_next': True, 
                 'masked_lm_positions': [4, 5, 18, 19, 20, 33, 55, 61, 69, 70, 71, 77, 90, 94, 97, 99, 121, 122, 123], 
                 'masked_lm_labels': ['阳', '区', '，', '行', '至', '于', '管', '分', '经', '审', '查', '，', '（', '份', '，', '籍', '，', '两', '人']}
                '''
                instances.append(instance)
            current_chunk = []  # 清空当前块
            current_length = 0  # 重置当前文本块的长度
        i += 1  # 接着文档中的内容往后看
    return instances


def create_training_instances(input_file, tokenizer, max_seq_len, short_seq_prob,
                              max_ngram, masked_lm_prob, max_predictions_per_seq):
    all_documents = [[]]
    f = open(input_file, 'r', encoding='utf8')
    lines = f.readlines()

    for line_cnt, line in enumerate(lines):
        line = line.strip()
        # Empty lines are used as document delimiters   # 文档直接使用空行进行分开的
        if not line:
            all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
            all_documents[-1].append(tokens)
    # print(all_documents[0])   # [[[文章1的句子1分词形式 是列表], [文章1的句子2分词形式]...], [[文章2的句子1分词形式], 文章2的句子2分词形式...]] ... ]
    # print(len(all_documents))
    all_documents = [x for x in all_documents if x]   # 移除空文本

    random.shuffle(all_documents)   # 将文章进行shuffle

    vocab_words = list(tokenizer.vocab.keys())   # 这里直接认为我们的词表就是bert-base的词表

    instances = []
    for document_index in range(len(all_documents)):
        instances.extend(
            create_instances_from_document(
                all_documents, document_index, max_seq_len, short_seq_prob,
                max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_words)
        )
    random.shuffle(instances)
    return instances


if __name__ == '__main__':
    # input_file = './corpus/pro_data.txt'
    tokenizer = BertTokenizer(vocab_file='./bert_pretrain/vocab.txt', do_lower_case=True)
    max_seq_len = 128
    short_seq_prob = 0.1
    max_ngram = 3
    masked_lm_prob = 0.15
    max_predictions_per_seq = 20

    file_list = ['./corpus/pro_data.txt', ]
    for i, input_file in enumerate(file_list):
        with open('process_data{}.json'.format(i), 'w', encoding='utf8') as fw:
            file_examples = create_training_instances(input_file, tokenizer, max_seq_len, short_seq_prob,
                                                      max_ngram, masked_lm_prob, max_predictions_per_seq)
            file_examples = [json.dumps(instance) for instance in file_examples]
            for instance in file_examples:
                fw.write(instance + '\n')



