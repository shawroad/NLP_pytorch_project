"""
@file   : create_data.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-08-10
"""
import pandas as pd
from transformers import BertTokenizer
import random
from tqdm import tqdm
from nltk import sent_tokenize
import numpy as np
import collections
import json
from transformers.models.bert import BertTokenizer


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            # 加起来都没最大的长度长，还截个毛线
            break
        # 看A和B  谁长截谁
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # 截断长的序列的时候，随机从前面截或者后面截
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def create_masked_lm_predictions(tokens, max_ngram, masked_lm_prob, max_predictions_per_seq, vocab_list):
    ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_ngram + 1)
    pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
    # print(pvals)  # array([0.54545455, 0.27272727, 0.18181818])  ngram=1概率微0.545, ngram=2 概率微0.2727, 等等。
    # 这里ngram等于几  相当于就是得到当前mask位置以后，看要不要再加几个token进来，对ngram进行mask
    # 以上三步是想办法对ngram进行mask

    # 得到可以进行mask的token列表
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indices.append(i)
    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    random.shuffle(cand_indices)

    masked_token_labels = []
    covered_indices = set()
    for index in cand_indices:
        n = np.random.choice(ngrams, p=pvals)   # ngram选多大  [1, 2, 3, max_ngram] 按概率选一个

        if len(masked_token_labels) >= num_to_mask:
            # mask的个数够了 则终止
            break
        if index in covered_indices:
            # 如果当前的index之前已经被选中mask了，则终止此次循环
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
    # document: [[文章第一句的分词形式], [文章第二句的分词形式], [文章第三句的分词形式],...]
    while i < len(document):  # 从文档的开始看起  一句话一句话的看
        segment = document[i]   # segment是列表，代表的是按字分开的一个完整句子  # 句子
        # segment = get_new_segment(segment)  # whole word mask for chinese: 结合分词的中文的whole mask设置即在需要的地方加上“##”
        current_chunk.append(segment)  # 将一个独立的句子加入到当前的文本块中
        current_length += len(segment)  # 累计到为止位置  长度是多长

        if i == len(document) - 1 or current_length >= target_seq_length:
            # 如果累计的序列长度达到了目标的长度，或当前走到了文档结尾==>构造并添加到“A[SEP]B“中的A和B中；
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:   # 如果有超过两个句子 那就随机选个终点 这样就可能不止包含一个句子 这里选择的时A[SEP]B中的A
                    a_end = random.randint(1, len(current_chunk) - 1)
                # 将当前文本段中选取出来的部分，赋值给A即tokens_a
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                # 构造“A[SEP]B“中的B部分(有一部分是正常的当前文档中的后半部;在原BERT的实现中一部分是随机的从另一个文档中选取的，）
                tokens_b = []    # 构造B部分 正样本 即从A后面这部分直接作为B
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
                    tokens=tokens, max_ngram=max_ngram, masked_lm_prob=masked_lm_prob,
                    max_predictions_per_seq=max_predictions_per_seq, vocab_list=vocab_words
                )

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

    df = pd.read_csv(input_file, sep='\t')
    df.columns = ['label', 'content']

    all_documents = []
    # 分句
    with open('../data/train.csv', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            _, text = line.split('\t')
            sentence_split = sent_tokenize(text)
            if len(sentence_split) <= 1:
                # 之所以这样 是因为如果一篇文章只有一个句子 或者它的长度很短 则给它找不到正样本
                continue
            all_documents.append(sent_tokenize(text))    # 将每篇文章分句
    # all_documents: [['xxx', 'xxx', 'xxx',...], ['xxx', 'xxx', 'xxx', ...], ...]
    # 分词
    for i in tqdm(range(len(all_documents))):
        for j in range(len(all_documents[i])):
            all_documents[i][j] = tokenizer.tokenize(all_documents[i][j])
    # all_documents: [[[文章1的第一个句子分词列表], [文章2的第二个句子分词列表], []...], [[], [], [],...], []]

    random.shuffle(all_documents)   # 将文章进行shuffle

    vocab_words = list(tokenizer.vocab.keys())   # 这里直接认为我们的词表就是bert-base的词表

    instances = []
    for document_index in range(len(all_documents)):
        instances.extend(
            create_instances_from_document(
                all_documents=all_documents, document_index=document_index, max_seq_length=max_seq_len,
                short_seq_prob=short_seq_prob, max_ngram=max_ngram, masked_lm_prob=masked_lm_prob,
                max_predictions_per_seq=max_predictions_per_seq, vocab_words=vocab_words
            )
        )
    random.shuffle(instances)
    return instances


if __name__ == '__main__':
    tokenizer = BertTokenizer(vocab_file='../bert_pretrain/vocab.txt', do_lower_case=True)
    max_seq_len = 128   # 最大长度  可以设置到512
    short_seq_prob = 0.1    # 有0.1的概率选择长度小于max_seq_len的文本
    max_ngram = 3
    masked_lm_prob = 0.15    # 每个序列最大能mask掉token的占比
    max_predictions_per_seq = 20    # 每个序列的最大能mask掉几个token

    # 本示例采用imdb数据集
    path = '../data/train.csv'

    with open('process_data.json', 'w', encoding='utf8') as f:
        file_examples = create_training_instances(
            input_file=path, tokenizer=tokenizer, max_seq_len=max_seq_len,
            short_seq_prob=short_seq_prob, max_ngram=max_ngram,
            masked_lm_prob=masked_lm_prob, max_predictions_per_seq=max_predictions_per_seq)

        file_examples = [json.dumps(instance) for instance in file_examples]
        for instance in file_examples:
            f.write(instance + '\n')



