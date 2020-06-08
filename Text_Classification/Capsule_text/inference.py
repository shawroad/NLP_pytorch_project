"""

@file  : inference.py

@author: xiaolu

@time  : 2020-06-08

"""
import torch
import json
import numpy as np
from capsule import Capsule_Main
from config import Config


if __name__ == '__main__':
    # model load
    print('model loading...')
    model = Capsule_Main()
    model.to(Config.device)
    model.load_state_dict(torch.load('./best_model.bin'))

    print('data loading...')
    data = json.load(open('./data/train.json', 'r'))
    sentence_ids = data['sentence_ids']
    labels = data['labels']
    lengths = data['sentence_len']

    # select one data
    d = sentence_ids[2]
    label = labels[2]
    with torch.no_grad():
        d = torch.LongTensor([d])
        logits = model(d)
        pred = np.argmax(logits.cpu().data.numpy(), axis=1)[0]
        print("true label:", label)
        print('pred label:', pred)
