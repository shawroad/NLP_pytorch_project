"""

@file  : inference_all.py

@author: xiaolu

@time  : 2020-04-14

"""
import torch
from DataLoader import DatasetIterater, build_dataset
from model import Model
from config import Config
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm


def predict():
    # 准备一条数据
    train_data, dev_data = build_dataset(Config)
    tokenizer = BertTokenizer.from_pretrained('./albert_pretrain/vocab.txt')

    # 加载模型
    model = Model().to(Config.device)
    model.load_state_dict(torch.load('./save_model/' + 'best_model.bin', map_location='cpu'))
    print("模型加载成功...")
    model.eval()

    Q = []
    P = []
    T = []
    for data in tqdm(train_data):
        ids, input_ids, input_mask, start_, end_ = data
        context = tokenizer.decode(input_ids)
        context = context.split(' ')
        true_answer = context[start_: end_]
        # print(start_, end_)

        input_ids = torch.LongTensor([input_ids])
        input_mask = torch.LongTensor([input_mask])

        with torch.no_grad():
            start, end = model(input_ids, attention_mask=input_mask)
            # start = start.data.cpu().numpy()
            # end = end.data.cpu().numpy()
            start = torch.max(start.data, 1)[1].cpu().numpy()
            end = torch.max(end.data, 1)[1].cpu().numpy()

            start = start.tolist()[0]
            end = end.tolist()[0]
            # print(start, end)

            if start < end:
                answer = context[start: end]
            else:
                answer = context[end: start]

            if start == end:
                answer = '很可惜, 没有答案'

        context = ''.join(context)
        question = context.split('[SEP]')[0].replace('[CLS]', '')
        pred_answer = ''.join(answer)
        true_answer = ''.join(true_answer)

        Q.append(question)
        P.append(pred_answer)
        T.append(true_answer)

    data = {
        'question': Q,
        'true_answer': T,
        'pred_answer': P
    }
    data = pd.DataFrame(data)
    data.to_csv('./result.csv')


if __name__ == '__main__':
    predict()
