"""

@file  : data_process.py

@author: xiaolu

@time  : 2020-01-03

"""
import glob
import pickle
from config import Config


def load_data(p):
    with open(p, 'r') as f:
        lines = f.readlines()
        content = []
        question = []
        answer = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue

            if 'q' in line:
                question.append(line)
            elif 'a' in line:
                answer.append(line)
            else:
                content.append(line)
    c = ''.join(content)  # 当前的文章
    input_corpus = []
    for q in question:
        c_q = c + q
        input_corpus.append(c_q)
    output_corpus = []
    for a in answer:
        output_corpus.append(a)

    return input_corpus, output_corpus


def build_vocab(total_corpus):
    vocab = list(set(list(total_corpus)))
    # 整理词表
    vocab2id = {}
    vocab2id['<pad>'] = 0
    vocab2id['<sos>'] = 1
    vocab2id['<eos>'] = 2
    vocab2id['<unk>'] = 3
    for i, v in enumerate(vocab):
        vocab2id[v] = i + 4

    id2vocab = {}
    for v, i in vocab2id.items():
        id2vocab[i] = v

    return vocab2id, id2vocab


def input_sentence_id(sentence, vocab2id):
    id_sent = []
    for s in sentence:
        id_list = [vocab2id.get(i, Config.unk_id) for i in s]
        id_sent.append(id_list)
    return id_sent


def output_sentence_id(sentence, vocab2id):
    id_sent = []
    for s in sentence:
        id_list = []
        id_list.append(Config.sos_id)
        id_list.extend([vocab2id.get(i, Config.unk_id) for i in s])
        id_list.append(Config.eos_id)
        id_sent.append(id_list)
    return id_sent


def sequence_to_text(text, src_idx2char):
    sentence = []
    for i in text:
        word = src_idx2char.get(i, 'unk')
        sentence.append(word)
    return sentence


if __name__ == '__main__':
    data_path = glob.glob('./data/*')
    # print(data_path)
    input_corpus = []
    output_corpus = []
    for p in data_path:
        c_q, a = load_data(p)
        input_corpus.extend(c_q)
        output_corpus.extend(a)

    total_corpus = ''.join(input_corpus) + ''.join(output_corpus)

    vocab2id, id2vocab = build_vocab(total_corpus)

    vocab_dict = {'vocab2id': vocab2id, 'id2vocab': id2vocab}

    with open(Config.vocab_file, 'wb') as file:
        pickle.dump(vocab_dict, file)

    # 将语料转为id
    input_id = input_sentence_id(input_corpus, vocab2id)
    output_id = output_sentence_id(output_corpus, vocab2id)

    data = {'input_corpus': input_id, 'output_corpus': output_id}

    with open(Config.data_file, 'wb') as file:
        pickle.dump(data, file)


