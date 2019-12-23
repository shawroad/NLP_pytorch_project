"""

@file  : make_dictionary.py

@author: xiaolu

@time  : 2019-11-20

"""
import os
'''获取字,然后整理字典'''

text_path = "lyric"
dictionary_path = "dictionary/dictionary.txt"

words = set()

for filename in os.listdir(text_path):
    with open(os.path.join(text_path, filename), 'r+', encoding='utf8') as file:
        word = file.read(1)
        while word:
            if word == '\n' or word == '\r' or word == ' ':
                pass
            else:
                words.add(word)
            word = file.read(1)


with open(dictionary_path, "w+", encoding="utf-8") as f:
    f.write("[START] [SEQ] [UNK] [PAD] [END] ")
    f.write(" ".join(words))
    f.flush()
