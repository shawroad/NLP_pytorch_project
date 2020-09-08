"""

@file  : run_squad.py

@author: xiaolu

@time  : 2020-03-03

"""
import json
import args
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from tokenization import BertTokenizer

random.seed(args.seed)


def read_squad_examples(zhidao_input_file, search_input_file, is_training=True):
    total, error = 0, 0
    examples = []
    with open(search_input_file, 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            source = json.loads(line.strip())
            if len(source['answer_spans']) == 0:
                # 不存在答案的语料扔掉
                continue

            if source['answers'] == []:
                # 没有答案
                continue

            if source['match_scores'][0] < 0.8:
                # 答案匹配得分较低
                continue

            if source['answer_spans'][0][1] > args.max_seq_length:
                # 答案的结束比文本的长度还大 扔掉
                continue

            docs_index = source['answer_docs'][0]
            start_id = source['answer_spans'][0][0]
            end_id = source['answer_spans'][0][1] + 1  ## !!!!!
            question_type = source['question_type']
            # print(docs_index)   # 0
            # print(start_id)   # 9
            # print(end_id)   # 53
            # print(question_type)   # DESCRIPTION
            # exit()

            passages = []
            try:
                # 答案所在文章的id
                answer_passage_idx = source['documents'][docs_index]['most_related_para']
            except:
                continue

            # 选取答案所在的那篇文章 然后取出文章 进一步取出最相关的段落
            doc_tokens = source['documents'][docs_index]['segmented_paragraphs'][answer_passage_idx]
            ques_len = len(source['documents'][docs_index]['segmented_title']) + 1
            doc_tokens = doc_tokens[ques_len:]  # 把标题截取后的文章形式
            start_id, end_id = start_id - ques_len, end_id - ques_len

            # 检测标注答案的起始和终止位置
            if start_id >= end_id or end_id > len(doc_tokens) or start_id >= len(doc_tokens):
                continue

            new_doc_tokens = ""
            for idx, token in enumerate(doc_tokens):
                if idx == start_id:
                    new_start_id = len(new_doc_tokens)
                    break
                new_doc_tokens = new_doc_tokens + token

            new_doc_tokens = "".join(doc_tokens)
            new_end_id = new_start_id + len(source['fake_answers'][0])  # 加上假答案的长度

            if source['fake_answers'][0] != "".join(new_doc_tokens[new_start_id: new_end_id]):
                continue

            if is_training:
                example = {
                    "qas_id": source['question_id'],
                    "question_text": source['question'].strip(),   # 记住 这里不是分词形式
                    "question_type": question_type,
                    "doc_tokens": new_doc_tokens.strip(),   # 这里也不是分词形式
                    "start_position": new_start_id,
                    "end_position": new_end_id}
                examples.append(example)

    with open(zhidao_input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            source = json.loads(line.strip())

            if len(source['answer_spans']) == 0:
                continue
            if source['answers'] == []:
                continue
            if source['match_scores'][0] < 0.8:
                continue
            if source['answer_spans'][0][1] > args.max_seq_length:
                continue
            docs_index = source['answer_docs'][0]

            start_id = source['answer_spans'][0][0]
            end_id = source['answer_spans'][0][1] + 1  ## !!!!!
            question_type = source['question_type']

            passages = []
            try:
                answer_passage_idx = source['documents'][docs_index]['most_related_para']
            except:
                continue

            doc_tokens = source['documents'][docs_index]['segmented_paragraphs'][answer_passage_idx]

            ques_len = len(source['documents'][docs_index]['segmented_title']) + 1
            doc_tokens = doc_tokens[ques_len:]
            start_id, end_id = start_id - ques_len, end_id - ques_len

            if start_id >= end_id or end_id > len(doc_tokens) or start_id >= len(doc_tokens):
                continue

            new_doc_tokens = ""
            for idx, token in enumerate(doc_tokens):
                if idx == start_id:
                    new_start_id = len(new_doc_tokens)
                    break
                new_doc_tokens = new_doc_tokens + token

            new_doc_tokens = "".join(doc_tokens)
            new_end_id = new_start_id + len(source['fake_answers'][0])

            if source['fake_answers'][0] != "".join(new_doc_tokens[new_start_id:new_end_id]):
                continue

            if is_training:
                new_end_id = new_end_id - 1
                example = {
                    "qas_id": source['question_id'],
                    "question_text": source['question'].strip(),
                    "question_type": question_type,
                    "doc_tokens": new_doc_tokens.strip(),
                    "start_position": new_start_id,
                    "end_position": new_end_id}

                examples.append(example)

    print("len(examples):", len(examples))  # len(examples): 1573
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    '''
    :param examples: 处理好的文本 以字典的格式给出
    :param tokenizer: bert分词器
    :param max_seq_length: 文章的长度
    :param max_query_length: 问题长度
    :return:
    '''
    features = []

    for example in tqdm(examples):

        # 问题
        query_tokens = list(example['question_text'])   # 直接进行的分字

        # 问题类型
        question_type = example['question_type']

        # 文章
        doc_tokens = example['doc_tokens']
        doc_tokens = doc_tokens.replace(u"“", u"\"")
        doc_tokens = doc_tokens.replace(u"”", u"\"")

        # 开始和结束标志
        start_position = example['start_position']
        end_position = example['end_position']

        # 问题长度过长  直接截断
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # 制造bert的输入
        tokens = []
        segment_ids = []

        tokens.append("[CLS]")  # 加入bert语料的分类标志
        segment_ids.append(0)

        # 加入了的CLS 所以答案的起始和结束需要加1
        start_position = start_position + 1
        end_position = end_position + 1

        # 把问题当做bert输入的前半部分　　　[CLS] 问题的id形式　[SEP] 文章的id形式 [SEP]
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
            start_position = start_position + 1
            end_position = end_position + 1

        tokens.append('[SEP]')
        segment_ids.append(0)
        start_position = start_position + 1
        end_position = end_position + 1

        for i in doc_tokens:
            tokens.append(i)
            segment_ids.append(1)
        tokens.append('[SEP]')
        segment_ids.append(1)

        if end_position >= max_seq_length:
            continue

        # 长度若比我们要求的长度长  则截断
        if len(tokens) > max_seq_length:
            tokens[max_seq_length - 1] = '[SEP]'
            # 将刚才组装的语料token转为bert指定的id映射
            input_ids = tokenizer.convert_tokens_to_ids(tokens[:max_seq_length])
            segment_ids = segment_ids[:max_seq_length]
        else:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 加入mask
        input_mask = [1] * len(input_ids)
        assert len(input_ids) == len(segment_ids)

        features.append(
            {"input_ids": input_ids,
             "input_mask": input_mask,
             "segment_ids": segment_ids,
             "start_position": start_position,
             "end_position": end_position}
        )

    # with open("./train.data", 'w', encoding="utf-8") as fout:
    with open("./dev.data", 'w', encoding='utf-8') as fout:
        for feature in features:
            fout.write(json.dumps(feature, ensure_ascii=False) + '\n')
    print("len(features):", len(features))
    return features


if __name__ == '__main__':
    # 1 将文章转为id用的
    tokenizer = BertTokenizer.from_pretrained('./vocab.txt', do_lower_case=True)

    # 2 生成训练数据集 train.data
    # examples = read_squad_examples(zhidao_input_file=args.zhidao_input_file,
    #                                search_input_file=args.search_input_file)
    #
    # features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
    #                                         max_seq_length=args.max_seq_length,
    #                                         max_query_length=args.max_query_length)

    # 3 生成验证数据， dev.data。记得注释掉生成训练数据的代码，并在196行将train.data改为dev.data
    examples = read_squad_examples(zhidao_input_file=args.dev_zhidao_input_file,
                                   search_input_file=args.dev_search_input_file)

    features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length,
                                            max_query_length=args.max_query_length)
