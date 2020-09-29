import json
import torch
from transformers import BertTokenizer
import gzip
import pickle
from tqdm import tqdm
from model import Model
from config import set_args


class RankExample(object):
    def __init__(self,
                 doc_id=None,
                 question_text=None,
                 question_type=None,
                 context=None,
                 neg_context_id=None,
                 neg_context=None,
                 answer=None,
                 label=None,
                 ):
        self.doc_id = doc_id
        self.question_text = question_text
        self.question_type = question_type
        self.context = context
        self.neg_context_id = neg_context_id
        self.neg_context = neg_context
        self.answer = answer
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "doc_id: %s" % (str(self.doc_id))
        s += ", question_text: %s" % (self.question_text)
        s += ", question_type: %s" % (self.question_type)
        s += ", context: %s" % (self.context)
        s += ", neg_context_id: %d" % (self.neg_context_id)
        s += ", neg_context: %s" % (self.neg_context)
        s += ", answer: %s" % (self.answer)
        s += ", label: %d" % (self.label)
        return s


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, scores=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.scores = scores

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "input_ids: %s" % (self.input_ids)
        s += ", input_mask: %s" % (self.input_mask)
        s += ", segment_ids: %s" % (self.segment_ids)
        s += ", label_id: %s" % (self.label_id)
        s += ", scores: %s" % (self.scores)
        return s


def convert_examples_to_features(item, max_seq_length, tokenizer):
    pos_context = item.context
    neg_context = item.neg_context
    question = item.question_text

    pos_encode = tokenizer.encode_plus(question, pos_context)
    neg_encode = tokenizer.encode_plus(question, neg_context)

    # positive data
    t = {}
    if len(pos_encode['input_ids']) > max_seq_length:
        t['input_ids'] = pos_encode['input_ids'][:max_seq_length]
        t['token_type_ids'] = pos_encode['token_type_ids'][:max_seq_length]
        t['attention_mask'] = pos_encode['attention_mask'][:max_seq_length]
    else:
        t['input_ids'] = pos_encode['input_ids'] + [0] * (max_seq_length - len(pos_encode['input_ids']))
        t['token_type_ids'] = pos_encode['token_type_ids'] + [0] * (max_seq_length - len(pos_encode['token_type_ids']))
        t['attention_mask'] = pos_encode['attention_mask'] + [0] * (max_seq_length - len(pos_encode['attention_mask']))
    pos = InputFeatures(input_ids=t['input_ids'], input_mask=t['attention_mask'],
                        segment_ids=t['token_type_ids'], label_id=1)

    # negative data
    t = {}
    if len(neg_encode['input_ids']) > max_seq_length:
        t['input_ids'] = neg_encode['input_ids'][:max_seq_length]
        t['token_type_ids'] = neg_encode['token_type_ids'][:max_seq_length]
        t['attention_mask'] = neg_encode['attention_mask'][:max_seq_length]
    else:
        t['input_ids'] = neg_encode['input_ids'] + [0] * (max_seq_length - len(neg_encode['input_ids']))
        t['token_type_ids'] = neg_encode['token_type_ids'] + [0] * (max_seq_length - len(neg_encode['token_type_ids']))
        t['attention_mask'] = neg_encode['attention_mask'] + [0] * (max_seq_length - len(neg_encode['attention_mask']))
        assert len(t['input_ids']) == len(t['token_type_ids']) == len(t['attention_mask'])
    neg = InputFeatures(input_ids=t['input_ids'], input_mask=t['attention_mask'],
                        segment_ids=t['token_type_ids'], label_id=0)
    return [pos, neg]


def calc_score(feature):
    input_ids = torch.tensor([feature.input_ids], dtype=torch.long)
    input_mask = torch.tensor([feature.input_mask], dtype=torch.long)
    segment_ids = torch.tensor([feature.segment_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=input_mask, segment_ids=segment_ids)
        return logits


if __name__ == "__main__":
    args = set_args()
    device = torch.device('cuda: {}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    # with gzip.open('./examples.pkl.gz', 'r') as f:
    #     data = pickle.load(f)
    #
    # # 对数据进行预处理
    # tokenizer = BertTokenizer.from_pretrained(args.vocab_file)
    # # print(len(data))    # 59681
    # max_seq_length = 512
    # features = []
    # for d in tqdm(data):
    #     temp = convert_examples_to_features(d, max_seq_length, tokenizer)
    #     features.extend(temp)

    # # 暂时的features
    # with gzip.open('temp_train_features.pkl.gz', 'wb') as f:
    #     pickle.dump(features, f)

    # 加载暂时的features
    with gzip.open('temp_train_features.pkl.gz', 'r') as f:
        features = pickle.load(f)

    # 开始推理
    model = Model()
    model.load_state_dict(torch.load('./save_model/best_pytorch_model.bin', map_location='cpu'))
    model.to(device)
    result = []
    i = 0
    for feature in tqdm(features):
        i += 1
        s = calc_score(feature)     # tensor([[-2.4370,  2.7352]])
        temp = s.tolist()[0]
        feature.scores = temp
        result.append(feature)

    with gzip.open('/train_features.pkl.gz', 'wb') as fout:
        pickle.dump(result, fout)

