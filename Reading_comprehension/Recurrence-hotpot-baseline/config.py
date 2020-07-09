# -*- coding: utf-8 -*-
# @Time    : 2020/7/8 17:26
# @Author  : xiaolu
# @FileName: config.py
# @Software: PyCharm
import torch


class Config:
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    train_data_path = './data/hotpot_train_v1.1.json'
    test_data_path = './data/hotpot_dev_v1.1.json'

    # 处理数据后生成的文件保存地址
    glove_word_file = './data/glove.840B.300d.txt'
    # glove_word_file = './data/word2idx.json'
    glove_word_size = 395456
    glove_dim = 300

    char_dim = 8

    char_hidden = 100
    hidden = 80

    train_eval_file = "./data/train_eval.json"
    dev_eval_file = "./data/dev_eval.json"
    test_eval_file = "./data/test_eval.json"

    train_record_file = './data/train_record.pkl'
    dev_record_file = './data/dev_record.pkl'
    test_record_file = './data/test_record.pkl'

    para_limit = 1000
    ques_limit = 80
    char_limit = 16
    sent_limit = 100

    word_emb_file = "./data/word_emb.json"
    char_emb_file = "./data/char_emb.json"
    word2idx_file = './data/word2idx.json'
    char2idx_file = './data/char2idx.json'
    idx2word_file = './data/idx2word.json'
    idx2char_file = './data/idx2char.json'

    batch_size = 2
    init_lr = 0.5
    keep_prob = 0.8

    sp_lambda = 0

    save = './save_model/'
    patience = 1

