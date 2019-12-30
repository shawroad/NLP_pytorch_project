"""

@file   : extract.py

@author : xiaolu

@time   : 2019-12-26

"""
import zipfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    extract('data/douban-multiturn-100w.zip')

