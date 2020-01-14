"""

@file  : data_process.py

@author: xiaolu

@time  : 2020-01-07

"""
from tqdm import tqdm
import spacy
from collections import Counter
import numpy as np
from codecs import open
import os
import json
from config import Config


nlp = spacy.blank("en")


def word_tokenize(sent):
    '''
    类似与分词
    :param sent:
    :return:
    '''
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    '''
    标注出每个词在文中出现的位置
    :param text:
    :param tokens:
    :return:
    '''
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print('Token {} cannot be found'.format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    '''
    对数据进行预处理
    :param filename:
    :param data_type:
    :param word_counter:
    :param char_counter:
    :return:
    '''
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, 'r') as f:
        source = json.load(f)
        for article in tqdm(source['data']):
            for para in article['paragraphs']:
                context = para['context'].replace("''", '" ').replace("``", '" ')    # 一篇文章
                context_tokens = word_tokenize(context)   # 相当于就是以空格进行分词

                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)  # 得到每个词在文中的起始和结束位置

                for token in context_tokens:
                    word_counter[token] += len(para['qas'])
                    for char in token:
                        char_counter[char] += len(para['qas'])

                # 取出问题
                for qa in para["qas"]:
                    total += 1
                    ques = qa['question'].replace("''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1

                    # 去答案
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa['answers']:
                        answer_text = answer['text']    # 答案文本
                        answer_start = answer['answer_start']    # 答案的起始标志
                        answer_end = answer_start + len(answer_text)  # 答案的结束标志

                        answer_texts.append(answer_text)
                        answer_span = []
                        # print(answer_start)   # 515
                        # print(answer_end)     # 541
                        for idx, span in enumerate(spans):  # 遍历文本中的每个单词的起始和结束标志
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)

                        y1, y2 = answer_span[0], answer_span[-1]   # 标记答案是当前文中的开始的第几个和结束的第几个
                        y1s.append(y1)
                        y2s.append(y2)

                    example = {"context_tokens": context_tokens, "context_chars": context_chars,
                               "ques_tokens": ques_tokens, "ques_chars": ques_chars,
                               "y1s": y1s, "y2s": y2s,
                               "id": total
                               }

                    examples.append(example)
                    eval_examples[str(total)] = {
                        'context': context, 'spans': spans, 'answers': answer_texts, 'uuid': qa['id']
                    }
        # print(examples)  # 单词组成的文本, 由字母组成的单词 单词组成的列表, 问题组成的文本, 问题字母组成, 答案起始, 答案结束, id
        # print(eval_examples)  # 英语文本, spans(每个单词在文中的起始和结束标志), 答案, 问题编号
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None):
    '''
    加载词向量
    :param counter: 统计一个列表中每个元素出现的次数　返回为一个字典
    :param data_type:　看是基于词还是基于字符
    :param limit:　是否低频次过滤掉
    :param emb_file:　预训练词向量文件位置
    :param vec_size: 词向量的维度
    :return:
    '''
    print("Generating {} embedding...".format(data_type))

    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]

    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, 'r', encoding='utf8') as f:  # 一个词对应一个向量　占一行
            for line in tqdm(f):
                array = line.split()
                word = ''.join(array[0: -vec_size])   # 词
                vector = list(map(float, array[-vec_size:]))  # 对应的向量

                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector

        print('{}/ {} tokens have corresponding {} embedding vector'.format(
            len(embedding_dict), len(filtered_elements), data_type
        ))
    else:
        # 没有词向量　我们就进行随机初始化
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    # 建立词典
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1

    # 对两个特殊标志进行编码
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]

    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}  # id->词向量

    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]   # 此向量矩阵
    # 返回一个词向量矩阵   词转id
    return emb_mat, token2idx_dict


def convert_to_features(config, data, word2idx_dict, char2idx_dict):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = config.para_limit
    ques_limit = config.ques_limit
    ans_limit = config.ans_limit
    char_limit = config.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
    y1 = np.zeros([para_limit], dtype=np.float32)
    y2 = np.zeros([para_limit], dtype=np.float32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def build_features(Config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    '''
    :param Config: 配置文件
    :param examples: 单词组成的文本, 由字母组成的单词 单词组成的列表, 问题组成的文本, 问题字母组成, 答案起始, 答案结束, id
    :param data_type: 看其是以字符还是单词
    :param out_file: 将整理的数据应该放在哪里
    :param word2idx_dict: 词转id
    :param char2idx_dict: 字符转id
    :param is_test: 是否是测试集
    :return:
    '''
    para_limit = Config.para_limit
    ques_limit = Config.ques_limit
    ans_limit = Config.ans_limit
    char_limit = Config.char_limit

    def filter_func(example, is_test=False):
        # 过滤不满足情况的句子
        return len(example['context_tokens']) > para_limit or len(example['ques_tokens']) > ques_limit or (example['y2s'][0] - example['y1s'][0]) > ans_limit

    print("Processing {} examples...".format(data_type))

    total = 0
    total_ = 0
    meta = {}
    N = len(examples)
    context_idxs = []   # 文本转为序列
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    for n, example in tqdm(enumerate(examples)):
        total_ += 1  # 显示文本的个数

        if filter_func(example, is_test):  # 返回True相当于不符合我们的标准
            continue

        total += 1   # 显示符合条件的文本

        def _get_word(word):
            # 将单词转id    单词, 将单词转小写, 将单词的首字母大写其他小写, 将单词转为大写 单词的这几种形式认为是一种
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            # 字符 　将字符转id
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        context_idx = np.zeros([para_limit], dtype=np.int32)   # 文章转为id序列
        context_char_idx = np.zeros([para_limit, char_limit])   # 一个单词一个序列　一个单词一个序列

        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        # 针对单词
        # 文章转id序列
        for i, token in enumerate(example['context_tokens']):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        # 问题转id序列
        for i, token in enumerate(example['ques_tokens']):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)

        # 针对字符
        # 文章中的字符转为id序列
        for i, token in enumerate(example['context_chars']):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        # 问题中的字符转为id序列
        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)

        # 开始标志
        start, end = example["y1s"][-1], example["y2s"][-1]
        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    # print(context_char_idxs)  # 文章的每个词有好几个字母组成　也就是一个词是一个id序列, 整片文章[[一个单词的字符id序列], [], []]
    # print(y2s)   # 结束列表

    np.savez(out_file, context_idxs=np.array(context_idxs), context_char_idxs=np.array(context_char_idxs),
             ques_idxs=np.array(ques_idxs), ques_char_idxs=np.array(ques_char_idxs), y1s=np.array(y1s),
             y2s=np.array(y2s), ids=np.array(ids))
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    '''
    保存
    '''
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def preproc():
    # 数据放在这个文件
    if not os.path.exists(Config.target_dir):
        os.makedirs(Config.target_dir)

    # 日志放在这个位置
    if not os.path.exists(Config.event_dir):
        os.makedirs(Config.event_dir)

    # 模型保存在这个位置
    if not os.path.exists(Config.save_dir):
        os.makedirs(Config.save_dir)

    # 答案
    if not os.path.exists(Config.answer_dir):
        os.makedirs(Config.answer_dir)

    # 对数据进行预处理
    word_counter, char_counter = Counter(), Counter()   # Counter()统计每个列表中每个元素出现的次数　返回为字典
    train_examples, train_eval = process_file(Config.train_file, "train", word_counter, char_counter)


    # train_examples数据集中的每一个元素格式(是一个字典):
    # {'context_tokens': [单词1, 单词2...], 'context_chars': [['h', 'o', 'w'], [...], ...],
    #  'ques_tokens': [单词1, 单词2...], 'ques_chars': [['W', 'h', 'a', 't'], [...], ...]},
    #  'y1s': [0], 'y2s': [2], 'id': 87599}

    # train_eval数据集中的每个元素格式(是一个字典):
    # {'context': "Kathmandu Metropolitan City (KMC), in order to promote..,
    #  'spans':  [(0, 9), (10, 22), (23, 27), (28, 29), (29. 32)...}
    #  'answers': ['Kathmandu Metropolitan City']
    #  'uuid': '5735d259012e2f140011a0a1'

    # 对验证数据进行预处理
    dev_examples, dev_eval = process_file(Config.dev_file, "dev", word_counter, char_counter)
    # test_examples, test_eval = process_file(config.test_file, "test", word_counter, char_counter)

    # 加载词向量
    word_emb_file = Config.fasttext_file if Config.fasttext else Config.glove_word_file  # 词嵌入
    char_emb_file = Config.glove_char_file if Config.pretrained_char else None   # 字符嵌入
    char_emb_dim = Config.glove_dim if Config.pretrained_char else Config.char_dim

    # 基于词
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, 'word', emb_file=word_emb_file, vec_size=Config.glove_dim
    )
    # print(word_emb_mat)  # [[词向量300维度], [词向量300维度], [词向量300维度], ...]
    # print(word2idx_dict)  # {'Kasthamandap': 91626, 'nunataks': 91627, ...., '--NULL--': 0, '--OOV--': 1}

    # 基于字符
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, vec_size=char_emb_dim)

    # print(char_emb_mat)  # [[字符嵌入后的向量], [字符嵌入后的向量], ...]
    # print(char2idx_dict)  # {'A': 2, 'r': 3, 'c': 4, 'h': 5, 'i': 6, ...}

    build_features(Config, train_examples, 'train', Config.train_record_file, word2idx_dict, char2idx_dict)

    dev_meta = build_features(Config, dev_examples, 'dev', Config.dev_record_file, word2idx_dict, char2idx_dict)

    # test_meta = build_features(config, test_examples, "test", config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)

    save(Config.word_emb_file, word_emb_mat, message='word embedding')
    save(Config.char_emb_file, char_emb_mat, message="char embedding")
    save(Config.train_eval_file, train_eval, message="train eval")
    save(Config.dev_eval_file, dev_eval, message="dev eval")

    # save(config.test_eval_file, test_eval, message="test eval")
    save(Config.word2idx_file, word2idx_dict, message="word dictionary")
    save(Config.char2idx_file, char2idx_dict, message="char dictionary")
    save(Config.dev_meta_file, dev_meta, message="dev meta")
    # save(config.test_meta_file, test_meta, message="test meta")
    print(dev_meta)

if __name__ == "__main__":
    preproc()