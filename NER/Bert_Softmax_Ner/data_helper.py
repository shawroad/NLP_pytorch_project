"""

@file  : data_helper.py

@author: xiaolu

@time  : 2020-05-25

"""
import os
import random


def load_dataset(path_dataset):
    '''
    加载数据集
    :param path_dataset: data path
    :return:
    '''
    dataset = []
    with open(path_dataset, 'r') as f:
        words, tags = [], []
        for line in f:
            if line != '\n':   # 如果等于'\n'代表当前句子结束
                line = line.strip('\n')
                word, tag = line.split('\t')  # 字, 标注
                if len(word) > 0 and len(tag) > 0:
                    word, tag = str(word), str(tag)
                    words.append(word)
                    tags.append(tag)
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
    return dataset


def save_dataset(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences, open(os.path.join(save_dir, 'tags.txt'), 'w') as file_tags:
        for words, tags in dataset:
            file_sentences.write('{}\n'.format(' '.join(words)))
            file_tags.write('{}\n'.format(' '.join(tags)))
    print("数据保存完毕")


def build_tags(data_dir, tags_file):
    '''
    建立标签集合
    :param data_dir:
    :param tags_file:
    :return:
    '''
    data_types = ['train', 'val', 'test']
    tags = set()
    for data_type in data_types:
        tags_path = os.path.join(data_dir, data_type, 'tags.txt')
        with open(tags_path, 'r') as file:
            for line in file:
                tag_seq = filter(len, line.strip().split(' '))
                tags.update(list(tag_seq))
    tags = list(tags)
    with open(tags_file, 'w') as file:
        file.write('\n'.join(tags))
    return tags


if __name__ == '__main__':
    path_train_val = './data/msra/msra_train_bio'
    path_test = './data/msra/msra_test_bio'

    # 加载数据集
    print('正在加载数据．．．')
    dataset_train_val = load_dataset(path_train_val)
    dataset_test = load_dataset(path_test)
    print('训练集+验证集总共条数:', len(dataset_train_val))    # 训练集+验证集总共条数: 45000
    print('测试集的总共条数:', len(dataset_test))    # 测试集的总共条数: 3442
    print('数据加载完毕．．．')

    order = list(range(len(dataset_train_val)))
    random.seed(2020)
    random.shuffle(order)   # 将数据集打乱

    # 切分训练集和验证集
    train_dataset = [dataset_train_val[idx] for idx in order[:42000]]
    val_dataset = [dataset_train_val[idx] for idx in order[42000:]]
    test_dataset = dataset_test

    # 保存数据
    save_dataset(train_dataset, 'data/msra/train')
    save_dataset(val_dataset, 'data/msra/val')
    save_dataset(test_dataset, 'data/msra/test')

    # Build tags from dataset
    build_tags('data/msra', 'data/msra/tags.txt')

