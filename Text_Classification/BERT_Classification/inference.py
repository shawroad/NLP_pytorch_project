"""

@file  : inference.py

@author: xiaolu

@time  : 2020-01-02

"""
import torch
from tqdm import tqdm
import time
import random
import torch.nn.functional as F
import bert


PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    '''
    加载数据
    :param config:
    :return:
    '''
    def load_dataset(path, pad_size=32):
        '''
        :param path: 数据路径
        :param pad_size: 想把文本padding成的尺寸
        :return:
        '''
        contents = []
        with open(path, 'r', encoding='utf8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, label = line.split('\t')
                token = config.tokenizer.tokenize(content)  # 转id
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
                # 返回内容包括: 文本转为id序列, 标签, 当前文本的真实长度, mask向量(padding部位填充为0 其余位置填充为1)
        return contents

    test = load_dataset(config.test_path, config.pad_size)
    return test


def predict(config, model):
    # 随机选取一条数据进行预测
    data = build_dataset(config)
    index = random.randint(0, len(data))
    select_data = data[index]   # 选中的数据
    x = torch.LongTensor([select_data[0]]).to(device)
    y = torch.LongTensor([select_data[1]]).to(device)
    seq_len = torch.LongTensor([select_data[2]]).to(device)
    mask = torch.LongTensor([select_data[3]]).to(device)

    inputs, labels = [x, seq_len, mask], y

    model.eval()

    with torch.no_grad():
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)

        labels = labels.data.cpu().numpy()
        pred = torch.max(outputs.data, 1)[1].cpu().numpy()
        print("真实标签:{}, 预测标签:{}".format(labels, pred))
        print("此时的损失:{}".format(loss))


if __name__ == "__main__":
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
    dataset = 'THUCNews'   # 数据集
    config = bert.Config(dataset)  # 导入数据集
    model = bert.Model(config).to(config.device)
    predict(config, model)
