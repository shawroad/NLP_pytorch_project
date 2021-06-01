"""
@file   : utils.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-27
"""
import os
import torch
import json
from tqdm import tqdm
import numpy as np
from config import set_args
from functools import partial
from multiprocessing import Pool, cpu_count
from transformers.data import DataProcessor
from torch.utils.data import TensorDataset
from transformers.models.bert.tokenization_bert import whitespace_tokenize


args = set_args()


class SquadExample:
    def __init__(self, qas_id, question_text, context_text, answer_text, start_position_character, answers=[], is_impossible=False,):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.is_impossible = is_impossible
        self.answers = answers
        self.start_position, self.end_position = 0, 0

        doc_tokens, char_to_word_offset = [], []
        prev_is_whitespace = True
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # print(context_text)   # 本 报 讯 10 月 24 日   # 这样标注的数据 起始和终止位置还包含空格字符
        # print(doc_tokens)    # ['本', '报', '讯', '10', '月', '24', '日'
        # print(char_to_word_offset)   # [-1, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6,

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: {}".format(self.qas_id)
        s += ", question_text: {}".format(self.question_text)
        s += ", context_text: {}".format(self.context_text)
        s += ", answer_text: {}".format(self.answer_text)
        if self.start_position:
            s += ", start_position: {}".format(self.start_position)
        if self.end_position:
            s += ", end_position: {}".format(self.end_position)
        if self.is_impossible:
            s += ", is_impossible: {}".format(self.is_impossible)
        return s


class SquadFeaturesOrig:
    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            cls_index,
            p_mask,
            example_index,
            unique_id,
            paragraph_len,
            token_is_max_context,
            tokens,
            token_to_orig_map,
            start_position,
            end_position,
            is_impossible,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "input_ids: {}".format(self.input_ids)
        s += ", attention_mask: {}".format(self.attention_mask)
        s += ", token_type_ids: {}".format(self.token_type_ids)
        s += ", cls_index: {}".format(self.cls_index)
        s += ", p_mask: {}".format(self.p_mask)
        s += ", example_index: {}".format(self.example_index)
        s += ", unique_id: {}".format(self.unique_id)
        s += ", paragraph_len: {}".format(self.paragraph_len)
        s += ", token_is_max_context: {}".format(self.token_is_max_context)
        s += ", tokens: {}".format(self.tokens)
        s += ", token_to_orig_map: {}".format(self.token_to_orig_map)
        s += ", start_position: {}".format(self.start_position)
        s += ", end_position: {}".format(self.end_position)
        s += ", is_impossible: {}".format(self.is_impossible)
        return s


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


class SquadProcessor(DataProcessor):
    train_file = None
    dev_file = None
    test_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]
            answer = None
            answer_start = None
        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]
        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))
        return examples

    def get_train_examples(self, filename=None):
        '''
        预处理 训练集
        '''
        with open(filename, "r", encoding="utf-8") as reader:
            input_data = json.load(reader)["data"][0]["paragraphs"]   # 只加载了第一个paragraphs
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, filename=None):
        '''
        预处理 验证集
        '''
        with open(filename, 'r', encoding='utf8') as reader:
            input_data = json.load(reader)['data'][0]['paragraphs']   # 只加载了第一个paragraphs
        return self._create_examples(input_data, "dev")

    def get_test_examples(self, filename=None):
        '''
        预处理 测试集
        '''
        with open(filename, "r", encoding="utf-8") as reader:
            input_data = json.load(reader)["data"][0]["paragraphs"]   # 只加载了第一个paragraphs
        return self._create_examples(input_data, "test")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == 'train'    # 是否是训练集
        examples = []
        for paragraph in tqdm(input_data):
            title = paragraph['title'] if 'title' in paragraph else ''
            if len(title) != 0:
                context_text = title + paragraph['context']   # 如果有title 将其和文章进行拼接
            else:
                context_text = paragraph['context']

            # 遍历问题
            for qa in paragraph['qas']:
                question_text = qa['question']

                if question_text == '':
                    continue    # 问题为空

                qas_id = qa['id']
                start_position_character = None
                answer_text = None
                answers = []

                # 看当前问题是否可回答
                if 'is_impossible' in qa:
                    is_impossible = bool(qa["is_impossible"])
                else:
                    is_impossible = False   # 没有这个属性 默认其是可回答的

                if not is_impossible and qa.get('answers', []) != []:
                    if is_training:
                        answer = qa['answers'][0]    # 对于多个答案 只看第一个答案
                        answer_text = answer['text']   # 答案文本
                        start_position_character = answer['answer_start'] + len(title)  # 答案起始位置
                    else:
                        answers = qa['answers']

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    is_impossible=is_impossible,
                    answers=answers
                )
                examples.append(example)
        return examples


class MyProcessor(SquadProcessor):
    train_file = "train.json"
    dev_file = "dev.json"
    test_file = "test1.json"


def squad_convert_examples_to_features_orig(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
    '''
    将example转为feature
    '''
    features = []
    for example in tqdm(examples):
        feature = squad_convert_example_to_features_orig(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=is_training
        )
        features.append(feature)

    new_features = []
    unique_id = 1000000000   # 按截断后的样本分id
    example_index = 0   # 按原始样本分id
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    # convert to tensors
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

    if not is_training:
        # 非训练数据  不加start  end
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
        )
    else:
        # 训练数据集
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_start_positions,
            all_end_positions,
            all_cls_index,
            all_p_mask,
            all_is_impossible,
        )
    return features, dataset


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    '''
    处理下面这种问题:
    Question: What country is the top exporter of electornics?
    Context: The Japanese electronics industry is the largest in the world
    Answer: Japan
    标注答案为Japan,但是标注的起始为1、结束为1.显然包含了Japanese,不够精确。因为上一步做了子词的处理
    这里将进一步更新起始和结束，使其更精确的表达出Japan
    '''
    tok_answer_text = ' '.join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = ' '.join(doc_tokens[new_start: new_end + 1])
            if text_span == tok_answer_text:
                return new_start, new_end
    return input_start, input_end


def squad_convert_example_to_features_orig(example, tokenizer, max_seq_length, doc_stride, max_query_length, is_training):
    single_sample_features = []
    if is_training and not example.is_impossible:
        start_position = example.start_position
        end_position = example.end_position
        # 检查标注答案的起始结束位置  和 答案文本是否能对上
        actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            print("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    # 考虑子词的影响
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    # eg: 你 是 非 常 handsome => 你 是 非 常 hand some
    # orig_to_tok_index: [0, 1, 2, 3, 5]
    # tok_to_orig_index: [0, 1, 2, 3, 4, 4]

    # 考虑子词后修正起始和结束位置
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []
    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)

    sequence_added_tokens = 2   # 一定是2
    sequence_pair_added_tokens = 3

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):
        encoded_dict = {}
        try:
            # '[CLS]' + question + '[SEP]' + context + '[SEP]'
            encoded_dict = tokenizer.encode_plus(
                truncated_query,
                span_doc_tokens,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                pad_to_max_length=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
                return_token_type_ids=True,
            )
        except:
            print(example.qas_id)

        # 此次截断的输入样本中 文章的的真实长度
        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

        if tokenizer.pad_token_id in encoded_dict['input_ids']:
            # 找到第一个padding的位置  然后将之前的id截出来 然后转为对应的文本
            non_padded_ids = encoded_dict['input_ids'][: encoded_dict['input_ids'].index(tokenizer.pad_token_id)]
        else:
            # 如果没有padding的id 说明当前序列中没有padding
            non_padded_ids = encoded_dict['input_ids']

        # 将去除padding的输入样本转为文本
        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]
        # token_to_orig_map: {29: 0, 30: 1, 31: 2, 32: 3, 33: 4, 34: 5, 35: 6, 36: 7, ...}

        encoded_dict['paragraph_len'] = paragraph_len
        encoded_dict['tokens'] = tokens
        encoded_dict['token_to_orig_map'] = token_to_orig_map

        # 下面是query加上两个无用字符
        encoded_dict['truncated_query_with_special_tokens_length'] = len(truncated_query) + sequence_added_tokens
        encoded_dict['token_is_max_context'] = {}
        encoded_dict['start'] = len(spans) * doc_stride
        encoded_dict['length'] = paragraph_len
        spans.append(encoded_dict)

        if len(encoded_dict['overflowing_tokens']) == 0:
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]['paragraph_len']):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            # print(is_max_context)  # True
            index = spans[doc_span_index]['truncated_query_with_special_tokens_length'] + j
            # print(index)   # 19
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context  # {19: True}

    for span in spans:
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)   # [CLS]

        # p_mask: 1 代表不能预测的token  0 代表就是在这些token中选起始和结束位置
        p_mask = np.array(span["token_type_ids"])
        p_mask = np.minimum(p_mask, 1)   #
        p_mask = 1 - p_mask   # 原本token_type_ids= [0,0,0,0,1,1,1,1,1] 用1减  则变为[1,1,1,1,0,0,0,0,0]

        p_mask[np.where(np.array(span['input_ids']) == tokenizer.sep_token_id)[0]] = 1  # 将输入序列中的sep也置为1 不进行预测

        p_mask[cls_index] = 0    # CLS置为0  预测是否可回答的

        span_is_impossible = example.is_impossible   # 对每个span先按整体的doc标注 对于每个span再看是否可回答
        start_position, end_position = 0, 0
        if is_training and not span_is_impossible:
            doc_start = span['start']
            doc_end = span['start'] + span['length'] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                # 答案没在这个span中
                out_of_span = True

            if out_of_span:
                # 如果答案没在当前span 则将start和end表位cls_index
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True

            else:
                doc_offset = len(truncated_query) + sequence_added_tokens
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        single_sample_features.append(
            SquadFeaturesOrig(
                input_ids=span['input_ids'],
                attention_mask=span['attention_mask'],
                token_type_ids=span['token_type_ids'],
                cls_index=cls_index,
                p_mask=p_mask.tolist(),
                example_index=0,
                unique_id=0,
                paragraph_len=span['paragraph_len'],
                token_is_max_context=span['token_is_max_context'],
                tokens=span['tokens'],
                token_to_orig_map=span['token_to_orig_map'],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
            )
        )
    return single_sample_features


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    '''
    Check if this is the 'max context' doc span for the token.
    '''
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index


class SquadResult(object):
    """
    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "unique_id: {}".format(self.unique_id)
        s += ", start_logits: {}".format(self.start_logits)
        s += ", end_logits: {}".format(self.end_logits)
        return s
