"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-04-13
"""
import jieba
import copy
import pandas as pd
import numpy as np
import torch
from torch import nn
from model import Model
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from data_process import BOW
from config import set_args


def load_embedding():
    embedding = {}
    with open('./data/embedding_all_fasttext2_300.txt', 'r', encoding='utf8') as f:
        for line in f.readlines()[1:]:
            line = line.strip().split(' ')
            embedding[line[0]] = [float(i) for i in line[1:]]
    return embedding


if __name__ == '__main__':
    args = set_args()
    # 加载数据
    data = pd.read_csv(args.train_data_path)
    data['content'] = data.content.map(lambda x: ''.join(x.strip().split()))  # 去掉换行等

    # 把主题和情感拼拼接起来 一共10*3类   原本是主题为十分类 情感是三分类 现在整成30分类
    data['label'] = data['subject'] + data['sentiment_value'].astype(str)
    subj_lst = list(filter(lambda x: x is not np.nan, list(set(data.label))))
    subj_lst_dict = {value: key for key, value in enumerate(subj_lst)}
    data['label'] = data['label'].apply(lambda x: subj_lst_dict.get(x))

    # 多标签
    data_tmp = data.groupby('content').agg({'label': lambda x: set(x)}).reset_index()
    mlb = MultiLabelBinarizer()    # 转为类似于one-hot
    data_tmp['hh'] = mlb.fit_transform(data_tmp.label).tolist()
    y_train = np.array(data_tmp.hh.tolist())

    # 构建embedding
    bow = BOW(data_tmp.content.apply(jieba.lcut).tolist(), min_count=1, maxlen=100)  # 长短补齐  固定长度为100
    vocab_size = len(bow.word2idx)
    # print(vocab_size)   # 19887
    # print(bow.word_count)    # 统计每个词出现的次数
    embedding_matrix = np.zeros((vocab_size + 1, 300))
    # 加载词向量
    embedding = load_embedding()
    for key, value in bow.word2idx.items():
        if key in embedding.keys():
            embedding_matrix[value] = embedding[key]
        else:
            embedding_matrix[value] = [0] * 300

    X_train = copy.deepcopy(bow.doc2num)
    y_train = copy.deepcopy(y_train)

    batch_size = args.batch_size
    label_tensor = torch.tensor(np.array(y_train), dtype=torch.float)
    content_tensor = torch.tensor(np.array(X_train), dtype=torch.long)
    dataset = TensorDataset(content_tensor, label_tensor)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    model = Model(args.use_pre_embed, embedding_matrix, vocab_size)
    if torch.cuda.is_available():
        model.cuda()

    loss_func = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_train_epochs):
        for step, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = loss_func(output, target)

            acc = accuracy_score(np.argmax(target.cpu().data.numpy(), axis=1), np.argmax(output.cpu().data.numpy(), axis=1))
            print('epoch: {}, step: {}, loss: {:10f}, accuracy:{}'.format(epoch, step, loss, acc))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
