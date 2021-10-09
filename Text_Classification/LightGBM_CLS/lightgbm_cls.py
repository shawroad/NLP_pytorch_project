"""
@file   : lightgbm_cls.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-10-09
"""
import random
import jieba
import lightgbm as lgb   # conda install lightgbm
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split


def data_processing(path, test_size=0.2):
    '''
    :param path: 数据路径
    :param test_size: 训练集和测试集切分比
    :return:
    '''
    data_list = []
    label_list = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip().split('_!_')
            if len(line) >= 5:
                strr = line[3] + line[4]
            else:
                strr = line[3]
            word_cut = jieba.cut(strr, cut_all=False)
            word_list = list(word_cut)   # 一句话分词的结果 ['京城', '最', '值得', '你', '来场', '文化', '之旅', '的', ...]
            data_list.append(word_list)
            label_list.append(line[2])   # ['news_culture']  # 标签名

    data_class_list = list(zip(data_list, label_list))
    random.shuffle(data_class_list)  # 将data_class_list乱序

    index = int(len(data_class_list) * test_size) + 1  # 训练集和测试集切分的索引值
    train_list = data_class_list[index:]  # 训练集
    test_list = data_class_list[:index]  # 测试集
    train_data_list, train_class_list = zip(*train_list)  # 训练集解压缩
    test_data_list, test_class_list = zip(*test_list)  # 测试集解压缩

    # 统计训练集词频
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩
    all_words_list = list(all_words_list)  # 转换成列表
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def load_stop_word(file):
    '''
    加载停用词表
    :param file: 停用词文件位置
    :return:
    '''
    words_set = set()
    with open(file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                words_set.add(word)
    return words_set


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    feature_words = []  # 特征列表
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words的维度为1000
            break
            # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words


def convert_features(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list  # 返回结果


if __name__ == '__main__':
    # 文本预处理
    all_words_list, train_data_list, test_data_list, train_label_list, test_label_list = data_processing('./toutiao.txt', test_size=0.2)

    # 加载停用词
    stopwords_set = load_stop_word('./stopwords_cn.txt')

    # 将标签映射为id
    id2class = ['news_finance', 'news_story', 'news_travel', 'news_edu', 'news_military', 'news_game', 'news_agriculture', 'news_house', 'news_sports', 'news_car', 'news_tech', 'stock', 'news_entertainment', 'news_culture', 'news_world']
    class2id = {}
    index = 0
    for i in id2class:
        class2id[i] = index
        index = index + 1
    train_label_list = [class2id[i] for i in train_label_list]
    test_label_list = [class2id[i] for i in test_label_list]

    test_accuracy_list = []
    feature_words = np.load("./feature_words.npy")
    feature_words = list(feature_words)
    # print(len(feature_words))   # 754  特征词

    train_feature_list, test_feature_list = convert_features(train_data_list, test_data_list, feature_words)
    # print(len(train_feature_list[0]))   # 754
    # print(train_label_list[0])   # 2

    n_class = len(id2class)
    train_X = train_feature_list
    train_y = train_label_list
    test_X, dev_X, test_y, dev_y = train_test_split(test_feature_list, test_label_list, test_size=0.2, random_state=43)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y =np.array(test_y)
    dev_X = np.array(dev_X)
    dev_y = np.array(dev_y)
    print("train_X shape:", train_X.shape)
    print("train_y shape:", train_y.shape)
    print("test_X shape:", test_X.shape)
    print("test_y shape:", test_y.shape)
    print("dev_X shape:", dev_X.shape)
    print("dev_y shape:", dev_y.shape)
    dtrain = lgb.Dataset(train_X, label=train_y)
    dvalid = lgb.Dataset(dev_X, label=dev_y)
    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'num_class': n_class,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1
    }
    clf = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        early_stopping_rounds=100,
        verbose_eval=100,
        # feval=f1_score_eval
    )

    clf.save_model("./lightgbm.txt")

    predict_y = clf.predict(dev_X, num_iteration=clf.best_iteration)
    # predict_y = [list(x).index(max(x)) for x in predict_y]
    predict_y = np.argmax(predict_y, axis=1)
    print("result dev：")
    # print(f1_score(predict_y, dev_y, average='macro'))
    print(classification_report(dev_y, predict_y, target_names=id2class))

    predict_y = clf.predict(train_X, num_iteration=clf.best_iteration)
    # predict_y = [list(x).index(max(x)) for x in predict_y]
    predict_y = np.argmax(predict_y, axis=1)
    print("result train：")
    # print(f1_score(predict_y, train_y, average='macro'))
    print(classification_report(train_y, predict_y, target_names=id2class))

    predict_y = clf.predict(test_X, num_iteration=clf.best_iteration)
    # predict_y = [list(x).index(max(x)) for x in predict_y]
    predict_y = np.argmax(predict_y, axis=1)
    print("result test：")
    # print(f1_score(predict_y,test_y,average='macro'))
    print(classification_report(test_y, predict_y, target_names=id2class))