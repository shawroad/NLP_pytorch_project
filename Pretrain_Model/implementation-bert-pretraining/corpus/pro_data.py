# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 10:06
# @Author  : xiaolu
# @FileName: pro_data.py
# @Software: PyCharm

if __name__ == '__main__':
    with open('./corpus.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        d = ''
        for line in lines:
            temp = line.split('。')
            d += '\n'.join(temp)
    with open('./pro_data.txt', 'w', encoding='utf8') as f:
        f.write(d)
    exit()



