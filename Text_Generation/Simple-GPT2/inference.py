"""
@file   : inference.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-05-11
"""
import torch
import random
from model import Model

if __name__ == '__main__':
    # 本文用的是随机解码   top-k的方法

    # 随机生成一个其实标志
    vocab = torch.tensor([[random.randint(0, 126)]])   # 随机从词表中选一个词
    position = torch.tensor([[random.randint(0, 200)]])

    # 加载模型
    model = Model()
    model.load_state_dict(torch.load('./output/pytorch_model.bin'))

    model.eval()

    for i in range(200):
        output = model(vocab, position)
        # print(output.size())   # torch.Size([1, 1, 1032])   # 输入的第一个词
        output = output[:, -1:]

        # 取出概率最大的前8个token
        value, index = torch.topk(output, 8, dim=-1)

        # print(value)   # 前8大的概率
        # print(index)   # 前8大的概率对应的词(指的是在词表中的位置)

        value, index = value[0], index[0]

        value_index = torch.multinomial(torch.softmax(value, dim=-1), 1)  # 按概率随机采样一个索引
        # print(value_index)   # 假设采样的是第4个值  则第四个值就作为此次的输出

        output = index[0][value_index]

        vocab = torch.cat([vocab, output], dim=-1)
        position = torch.tensor([range(i + 2)])

    # 加载词典 将id转为汉字
    with open('./data/vocab.txt', 'r', encoding='utf8') as f:
        strs = f.read().split()

        for index in vocab[0]:
            if strs[index] == '[SEQ]':
                print()
            elif strs[index] == '[PAD]':
                print(' ', end='')
            elif strs[index] == '[START]':
                print()
            elif strs[index] == '[END]':
                print('end...')
                break
            else:
                print(strs[index], end="")


