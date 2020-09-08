"""

@file  : config.py

@author: xiaolu

@time  : 2020-01-07

"""
import os
import torch


class Config:
    # 指定设备
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    # 三种数据集
    # train_file = os.path.join("data", "squad", "train-v1.1.json")
    # dev_file = os.path.join("data", "squad", "dev-v1.1.json")
    train_file = './data/train-v1.1.json'
    dev_file = './data/dev-v1.1.json'
    test_file = os.path.join("data", "squad", "dev-v1.1.json")

    target_dir = "data"  # 数据
    event_dir = "log"   # 日志
    save_dir = "model"  # 模型保存的位置
    answer_dir = "log"

    # 词向量
    glove_word_file = os.path.join("data", "glove", "glove.840B.300d.txt")  # 词嵌入
    glove_char_file = os.path.join("data", "glove", "glove.840B.300d-char.txt")  # 字符嵌入
    glove_dim = 300  # Embedding dimension for Glove
    glove_char_size = 94  # Corpus size for Glove
    glove_word_size = int(2.2e6)  # Corpus size for Glove
    char_dim = 64  # Embedding dimension for char
    pretrained_char = False  # Whether to use pretrained char embedding

    fasttext_file = os.path.join("data", "fasttext", "wiki-news-300d-1M.vec")
    fasttext = False  # Whether to use fasttext

    # 文章, 问题, 答案长度的限制
    para_limit = 400  # Limit length for paragraph
    ques_limit = 50  # Limit length for question
    ans_limit = 30  # Limit length for answers
    char_limit = 16  # Limit length for character
    word_count_limit = -1  # Min count for word
    char_count_limit = -1  # Min count for char

    # 数据整理好 应该放的位置
    word_emb_file = os.path.join(target_dir, "word_emb.json")   # 词向量矩阵
    char_emb_file = os.path.join(target_dir, "char_emb.json")   # 字符向量矩阵

    train_eval_file = os.path.join(target_dir, "train_eval.json")  # 训练集
    dev_eval_file = os.path.join(target_dir, "dev_eval.json")   # 验证集

    train_record_file = os.path.join(target_dir, "train.npz")
    dev_record_file = os.path.join(target_dir, "dev.npz")
    test_record_file = os.path.join(target_dir, "test.npz")

    word2idx_file = os.path.join(target_dir, "word2idx.json")
    char2idx_file = os.path.join(target_dir, "char2idx.json")
    dev_meta_file = os.path.join(target_dir, "dev_meta.json")

    # 训练参数
    batch_size = 2  # Batch size
    num_steps = 60000  # Number of steps
    test_num_batches = 50  # Number of batches to evaluate the model

    # 模型参数
    learning_rate = 0.001
    lr_warm_up_num = 1000   # 1000步 学习率衰减一次
    ema_decay = 0.9999  # Exponential moving average decay
    grad_clip = 5.0  # Global Norm gradient clipping rate 梯度裁剪

    # 优化器中的参数
    beta1 = 0.8  # Beta 1
    beta2 = 0.999  # Beta 2

    checkpoint = 200  # checkpoint to save and evaluate the model

    d_model = 96  # Dimension of connectors of each layer
    num_heads = 8  # Number of heads in multi-head attention
    dropout = 0.1  # Dropout prob across the layers
    dropout_char = 0.05   # Dropout prob across the layers

    val_num_batches = 150  # Number of batches to evaluate the model

    n_head = 8
    vocab_size = 2925
    embedding_dim = 512

