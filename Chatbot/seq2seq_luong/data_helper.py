"""

@file  : data_helper.py

@author: xiaolu

@time  : 2020-04-01

"""
import unicodedata
import re
import torch
import itertools
import random
from config import Config


# 定义标志位置
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self):
        self.trimmed = False   # 是否去低频次
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # 这个放的是词表大小　目前只有三个标志位

    # addSentence() and addWord()　to build vocab_size and vocab
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        # 打印筛选出来的词占所有词的比例
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # 筛选完后　接下来构造词表
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)


def unicodeToAscii(s):
    # Turn a Unicode string to plain ASCII, thanks to
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def readVocs(corpus_path):
    '''
    加载语料
    :param corpus_path:
    :return:
    '''
    lines = open(corpus_path, encoding='utf8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc()
    return voc, pairs


def filterPair(p):
    '''
    过滤太长的句子
    :param p:
    :return:
    '''
    return len(p[0].split(' ')) < Config.MAX_LENGTH and len(p[1].split(' ')) < Config.MAX_LENGTH


def filterPairs(pairs):
    '''
    过滤句子
    :param pairs:
    :return:
    '''
    return [pair for pair in pairs if filterPair(pair)]


def trimRareWords(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)
    return keep_pairs


def loadPrepareData(data_path):
    print('清洗数据第一步:清洗数据集中乱码字符')
    voc, pairs = readVocs(data_path)
    print('清洗数据第二步:过滤过长的句子. ----', end='')
    pairs = filterPairs(pairs)
    print('语料总共为{}条'.format(len(pairs)))

    print("构建字典...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

# #######以上数据清晰完毕##########


# #######接下来准备把数据转id 并进行padding###########
def indexesFromSentence(voc, sentence):
    '''
    将一个句子转为id序列 并加结束标志
    :param voc:
    :param sentence:
    :return:
    '''
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    '''
    对句子进行padding
    :param l:
    :param fillvalue:
    :return:
    '''
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    '''
    输出数据(标签)准备mask向量
    :param l:
    :param value:
    :return:
    '''
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVar(l, voc):
    '''
    准备输入句子
    :param l:
    :param voc:
    :return:
    '''
    # 将句子转id  并加[EOS] ?
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    # 保存每个句子的真实长度
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    # 对句子进行padding   助理这里的填充将句子的维度　batch_size x seq_length 转为 seq_length x batch_size
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def outputVar(l, voc):
    '''
    准备输出数据(标签)
    :param l:
    :param voc:
    :return:
    '''
    # 将句子转id 这里加了EOS标识
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]

    # 统计输出数据中句子的最大长度
    max_target_len = max([len(indexes) for indexes in indexes_batch])

    # 对句子进行padding
    padList = zeroPadding(indexes_batch)

    # 准备mask句子
    mask = binaryMatrix(padList)

    # 将mask准备bool
    mask = torch.BoolTensor(mask)

    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(voc, pair_batch):
    # 将所有数据按照输出数据的长到短进行排序
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)

    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    # print(input_batch[:2])   # 输入的两个句子文本
    # print(output_batch[:2])   # 输出的两个句子文本

    # 准备input_data  将输入数据转id 并统计输入数据的所有长度
    inp, lengths = inputVar(input_batch, voc)

    # 准备output_data 将输出数据转为id 并准备mask
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


if __name__ == '__main__':
    data_path = './data/chatbot.txt'
    voc, pairs = loadPrepareData(data_path)

    # 把含有低频词的句子扔掉
    MIN_COUNT = Config.MIN_COUNT
    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    # print(pairs[:2])

    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(Config.batch_size)])]

