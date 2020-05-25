"""

@file  : inference.py

@author: xiaolu

@time  : 2020-05-25

"""
import torch
from transformers import BertTokenizer
from model import Model
from config import Config


if __name__ == '__main__':
    # 1. 准备数据
    input_sentence = '中华人民共和国是世界最屌的民族, 中国银行, 哈哈西安电子科技大学'
    tokenizer = BertTokenizer.from_pretrained(Config.model_vocab_path)
    tokens = tokenizer.tokenize(input_sentence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.LongTensor([input_ids])
    batch_masks = input_ids.gt(0)

    # 2. 加载标签
    id2tag = {}
    with open('./data/msra/tags.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            id2tag[i] = line

    # 2. 加载模型
    model = Model().to(Config.device)
    model.load_state_dict(torch.load('./save_model/' + 'best_model.bin', map_location='cpu'))
    print("模型加载成功...")
    model.eval()

    with torch.no_grad():
        bert_encode = model(input_ids, attention_mask=batch_masks)
        predicts = model.predict(bert_encode, batch_masks)
        predicts = predicts.numpy().tolist()
        result = [id2tag[i] for i in predicts]
        print(' '.join(result))


        

    # labels = torch.max(logits.data, 1)[1].cpu().numpy()
    # tags = [id2tag[i] for i in labels]
    # print(' '.join(tags))
