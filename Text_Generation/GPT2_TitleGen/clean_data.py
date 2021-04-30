"""
@file   : clean_data.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-04-30
"""
import re
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import random


def clean_weibo_title(title: str):
    '''
    对标题进行清洗
    :param title: 标题
    :return:
    '''

    # 去除##符号（一般为微博数据的话题标记）
    title = re.sub(r"#", "", title)

    # 去除[]中间的文字（一般为微博数据中的表情）
    title = re.sub(r"(\[{1,2})(.*?)(\]{1,2})", "", title)

    # 合并标题中过多的空格
    title = re.sub(r"\s+", " ", title)
    return title


def clean_weibo_content(content: str):
    '''
    对文章进行清洗
    :param content: 文章内容
    :return:
    '''

    # 去除网址
    content = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", content)

    # 合并正文中过多的空格
    content = re.sub(r"\s+", " ", content)

    # 去除\u200b字符
    content = content.replace("\u200b", "")
    return content


def clean_data(sample):
    '''
    整体清洗函数，为了方便多线程使用
    :param sample: 一个元组，包含正文内容和标题内容
    :return:
    '''
    (content, title) = sample
    sample = dict()
    # 清洗数据
    sample["title"] = clean_weibo_title(title.strip())
    sample["content"] = clean_weibo_content(content.strip())
    return sample


def build_news_data(content_path, title_path, train_save_path, test_save_path):
    '''
    加载数据并进行数据预处理 最后切分成训练集和测试集
    :param content_path:
    :param title_path:
    :param train_save_path:
    :param test_save_path:
    :return:
    '''
    content_data = open(content_path, 'r', encoding='utf8').readlines()
    title_data = open(title_path, 'r', encoding='utf8').readlines()

    data = zip(content_data, title_data)

    # 使用多线程处理数据
    threads = min(8, cpu_count())
    with Pool(threads) as p:
        annoate_ = partial(clean_data)
        data = list(
            tqdm(p.imap(annoate_, data, chunksize=8), desc='build data')
        )

    # 对数据进行过滤，去除重复数据、正文内容字长小于100的数据和标题内容字长小于100的数据
    data_set = set()
    data_new = []

    for d in data:
        if d['content'] in data_set or len(d['content']) < 100 or len(d['title']) < 2:
            continue
        else:
            data_set.add(d['content'])
            data_new.append(d)

    # 分割数据，构建训练集和测试集
    random.shuffle(data_new)
    train_data = data_new[:-3000]
    test_data = data_new[-3000:]
    fin = open(train_save_path, "w", encoding="utf-8")
    fin.write(json.dumps(train_data, indent=4, ensure_ascii=False))
    fin.close()
    fin = open(test_save_path, "w", encoding="utf-8")
    fin.write(json.dumps(test_data, indent=4, ensure_ascii=False))
    fin.close()


if __name__ == '__main__':
    content_path = './data/train_text.txt'
    title_path = './data/train_label.txt'

    train_save_path = './data/train_data.json'
    test_save_path = './data/test_data.json'

    build_news_data(content_path, title_path, train_save_path, test_save_path)
