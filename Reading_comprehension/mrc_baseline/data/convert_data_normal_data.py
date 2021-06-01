"""
@file   : convert_data_normal_data.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-31
"""
import json
from transformers import BasicTokenizer


def load_data(path, save_path):
    result = {}
    result['data'] = []
    tokenzier = BasicTokenizer(do_lower_case=True)

    corpus = json.load(open(path, 'r', encoding='utf8'))
    temp = {}
    for _ in corpus['data']:
        temp['title'] = _['title']
        temp['paragraphs'] = []

        for item in _['paragraphs']:
            item_dict = {}
            item_dict['context'] = ' '.join(tokenzier.tokenize(item['context']))
            item_dict['title'] =  ' '.join(tokenzier.tokenize(item['title']))
            item_dict['qas'] = []

            for ques in item['qas']:
                ques_dict = {}
                ques_dict['type'] = ques['type']
                ques_dict['question'] = ' '.join(tokenzier.tokenize(ques['question']))
                ques_dict['id'] = ques['id']
                ques_dict['answers'] = []
                ques_dict['is_impossible'] = ques['is_impossible']
                for ans in ques['answers']:
                    ans_dict = {}
                    ans_dict['text'] = ' '.join(tokenzier.tokenize(ans['text']))
                    ans_dict['answer_start'] = item_dict['context'].find(ans_dict['text'])
                    ques_dict['answers'].append(ans_dict)
                item_dict['qas'].append(ques_dict)
            temp['paragraphs'].append(item_dict)
        result['data'].append(temp)
    with open(save_path, 'w', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == '__main__':
    train_data_path = './train.json'
    save_train_data_path = 'train_process.json'

    dev_data_path = './dev.json'
    save_dev_data_path = 'dev_process.json'

    test_data_path = './test1.json'
    save_test_data_path = 'test_process.json'

    load_data(train_data_path, save_train_data_path)
    load_data(dev_data_path, save_dev_data_path)
    load_data(test_data_path, save_test_data_path)
