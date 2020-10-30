# -*- coding: utf-8 -*-
"""
@Time ： 2020/10/30 10:18
@Auth ： xiaolu
@File ：data_process.py
@IDE ：PyCharm
@Email：luxiaonlp@163.com
"""
import os
import copy
import json
import argparse
import torch
from tqdm import tqdm
from transformers import BertTokenizer
import gzip, pickle
from torch.utils.data import TensorDataset


class InputExample:
    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "guid: %s" % (str(self.guid))
        s += ", words: %s" % (self.words)
        s += ", intent_label: %s" % (self.intent_label)
        s += ", slot_labels: %s" % (self.slot_labels)
        return s


class InputFeatures:
    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "input_ids: %s" % (str(self.input_ids))
        s += ", attention_mask: %s" % (self.attention_mask)
        s += ", token_type_ids: %s" % (self.token_type_ids)
        s += ", intent_label_id: %s" % (self.intent_label_id)
        s += ", slot_labels_ids: %s" % (self.slot_labels_ids)
        return s


def read_file(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line.strip())
        return lines


def get_intent_labels(args):
    # 得到意图标签
    labels = []
    with open(os.path.join(args.data_dir, 'intent_label.txt'), 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            labels.append(line.strip())
    return labels


def get_slot_labels(args):
    # 槽标签
    labels = []
    with open(os.path.join(args.data_dir, 'slot_label.txt'), 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            labels.append(line.strip())
    return labels


def convert_examples_to_features(examples, max_seq_len, tokenizer, pad_token_label_id=-100, cls_token_segment_id=0,
                                 pad_token_segment_id=0, sequence_a_segment_id=0, mask_padding_with_zero=True):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for ex_index, example in tqdm(enumerate(examples)):
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            tokens.extend(word_tokens)
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len)

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            print("*** Example ***")
            print("guid: %s" % example.guid)
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            print("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            print("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids
                          ))

    return features


def load_data_to_examples(base_path, tokenizer, mode, intent_labels, slot_labels):
    base_path = base_path + '/' + mode
    input_text_file, intent_label_file, slot_labels_file = 'seq.in', 'label', 'seq.out'
    # 加载各种数据
    texts = read_file(os.path.join(base_path, input_text_file))
    intents = read_file(os.path.join(base_path, intent_label_file))
    slots = read_file(os.path.join(base_path, slot_labels_file))

    examples = []
    for i, (text, intent, slot) in tqdm(enumerate(zip(texts, intents, slots))):
        guid = "{}-{}".format(mode, i)

        # 1. input_text
        words = text.split()  # Some are spaced twice

        # 2. intent
        intent_label = intent_labels.index(intent) if intent in intent_labels else intent_labels.index('UNK')

        slot_label = []
        for s in slot.split():
            slot_label.append(slot_labels.index(s) if s in slot_labels else slot_labels.index('UNK'))
        assert len(words) == len(slot_label)
        examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_label))
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default='./bert_pretrain', required=False, type=str)
    parser.add_argument("--data_dir", default="./data/atis", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--mode", default='train', type=str)
    parser.add_argument("--save_examples", default='_examples.pkl.gz', type=str)
    parser.add_argument("--save_features", default='_features.pkl.gz', type=str)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.vocab)

    intent_label = get_intent_labels(args)
    slot_label = get_slot_labels(args)

    examples = load_data_to_examples(args.data_dir, tokenizer, args.mode, intent_label, slot_label)
    save_examples = './processed_data/' + args.mode + args.save_examples
    with gzip.open(save_examples, 'wb') as fout:
        pickle.dump(examples, fout)

    pad_token_label_id = 0
    features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id)
    save_features = './processed_data/' + args.mode + args.save_features
    with gzip.open(save_features, 'wb') as fout:
        pickle.dump(features, fout)







