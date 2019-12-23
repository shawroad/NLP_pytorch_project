"""

@file  : get_dictionary_length.py

@author: xiaolu

@time  : 2019-11-20

"""
# 看一下字典的大小
dictionary_path = "dictionary/dictionary.txt"
with open(dictionary_path, "r+", encoding="utf-8") as file:
    strs = file.read().strip().split()
    print(len(strs))
