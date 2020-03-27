"""

@file  : interface.py

@author: xiaolu

@time  : 2020-03-25

"""
import torch
from seq2seq_bert import Seq2SeqModel
from bert_model import BertConfig
from tokenizer import load_bert_vocab


if __name__ == '__main__':
    word2idx = load_bert_vocab()
    config = BertConfig(len(word2idx))
    bert_seq2seq = Seq2SeqModel(config)

    # 加载模型
    checkpoint = torch.load('./bert_dream.bin', map_location=torch.device("cpu"))
    bert_seq2seq.load_state_dict(checkpoint)
    bert_seq2seq.eval()

    test_data = ["梦见大街上人群涌动、拥拥而行的景象", "梦见司机将你送到目的地", "梦见别人把弓箭作为礼物送给自己", "梦见中奖了", "梦见大富豪"]
    for text in test_data:
        print(bert_seq2seq.generate(text, beam_size=3))

    print(bert_seq2seq.generate("1", beam_size=3))