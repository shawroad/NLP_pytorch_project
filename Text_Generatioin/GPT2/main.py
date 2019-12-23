"""

@file  : main.py

@author: xiaolu

@time  : 2019-11-20

"""
from trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer("models/net_lyric.pth", "data/tokenized/lyric/")
    trainer.train()
