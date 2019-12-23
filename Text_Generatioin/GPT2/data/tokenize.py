"""

@file  : tokenize.py

@author: xiaolu

@time  : 2019-11-20

"""
import os

'''把一篇文章的所有字编码'''

text_path = "lyric"
tokenized_path = "tokenized/lyric"
dictionary_path = "dictionary/dictionary.txt"

if not os.path.exists(tokenized_path):
    os.makedirs(tokenized_path)

with open(dictionary_path, "r+", encoding="utf-8") as file:
    dics = file.read().strip().split()

count = 0
for filename in os.listdir(text_path):
    f_path = os.path.join(text_path, filename)
    with open(f_path, "r+", encoding="utf-8") as f:
        indexs = ["0"]
        word = f.read(1)
        while word:
            if word == '\n' or word == '\r' or word == '\t' or ord(word) == 12288:
                indexs.append("1")
            elif word == ' ':
                indexs.append("3")
            else:
                try:
                    indexs.append(str(dics.index(word)))
                except:
                    indexs.append("2")

            word = f.read(1)
        count += 1

    with open(os.path.join(tokenized_path, "{}.txt".format(count)), "w+", encoding="utf-8") as df:
        df.write(" ".join(indexs))
