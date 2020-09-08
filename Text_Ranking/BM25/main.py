# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 10:31
# @Author  : xiaolu
# @FileName: main.py
# @Software: PyCharm
import operator
import re
from rank import score_BM25


class QueryParser:
    # 加载问题
    def __init__(self, filename):
        self.filename = filename
        self.queries = []

    def parse(self):
        with open(self.filename) as f:
            lines = ''.join(f.readlines())
        self.queries = [x.rstrip().split() for x in lines.split('\n')[:-1]]

    def get_queries(self):
        return self.queries


class CorpusParser:
    # 加载文章
    def __init__(self, filename):
        self.filename = filename
        self.regex = re.compile('^#\s*\d+')
        self.corpus = dict()

    def parse(self):
        with open(self.filename) as f:
            s = ''.join(f.readlines())   # 全部文章

        blobs = s.split('#')[1:]  # 第一堆多篇文章
        for x in blobs:
            text = x.split()   # 第一篇文章
            docid = text.pop(0)
            self.corpus[docid] = text   # {id1:文章, id2:文章, id3:文章...}

    def get_corpus(self):
        return self.corpus


def build_data_structures(corpus):
    # corpus的样子: {'1': [词1, 词2....], '2': [词1, 词2....], '3': [词1, 词2....]...}
    idx = InvertedIndex()
    dlt = DocumentLengthTable()
    for docid in corpus:  # 遍历文章id
        for word in corpus[docid]:  # 遍历当前文章的每个词
            idx.add(str(word), str(docid))  # 词,文章id
        length = len(corpus[str(docid)])  # 每篇文章中词的个数
        dlt.add(docid, length)
    # print(idx['extract'])
    # {'2': 1, '23': 1, '656': 1, '1043': 3, '1108': 1, '1309': 2} extract这个词在第2篇文章出现1次 在第23篇文章中出现1次...
    # print(dlt.table[str(2)])  # 31  相当于就是第二篇文章中有31个词
    return idx, dlt


class QueryProcessor:
    def __init__(self, queries, corpus):
        self.queries = queries
        self.index, self.dlt = build_data_structures(corpus)

    def run(self):
        results = []
        for query in self.queries:   # 遍历问题
            # print(self.run_query(query))   # {'1930': 9.646567300695821, '2246': 11.75268877838768, ...}
            results.append(self.run_query(query))
        return results

    def run_query(self, query):
        query_result = dict()
        for term in query:   # 遍历问题中的每个词
            if term in self.index:
                doc_dict = self.index[term]   # 取出当前词的在每篇文章中的统计次数
                for docid, freq in doc_dict.items():
                    # 文章id  当前问题中的这个词在当前文章中出现的次数
                    score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt),
                                       dl=self.dlt.get_length(docid),
                                       avdl=self.dlt.get_average_length())  # calculate score)
                    if docid in query_result:
                        query_result[docid] += score
                    else:
                        query_result[docid] = score
        return query_result


class InvertedIndex:
    def __init__(self):
        self.index = dict()

    def __contains__(self, item):
        return item in self.index

    def __getitem__(self, item):
        return self.index[item]

    def add(self, word, docid):
        # index: {词1:{1: freq_num, 2: freq_num, 3: freq_num, 4, 5, 6, 7, 8..}, {词2:{...}}}  # 当前这个词在每篇文章出现的次数
        if word in self.index:  # 如果当前词在字典中
            if docid in self.index[word]:   # 如果文章id也在这个词对应的列表中
                self.index[word][docid] += 1
            else:
                self.index[word][docid] = 1
        else:
            d = dict()
            d[docid] = 1
            self.index[word] = d

    # frequency of word in document
    def get_document_frequency(self, word, docid):
        if word in self.index:
            if docid in self.index[word]:
                return self.index[word][docid]
            else:
                raise LookupError('%s not in document %s' % (str(word), str(docid)))
        else:
            raise LookupError('%s not in index' % str(word))

    # frequency of word in index, i.e. number of documents that contain word
    def get_index_frequency(self, word):
        if word in self.index:
            return len(self.index[word])
        else:
            raise LookupError('%s not in index' % word)


class DocumentLengthTable:
    def __init__(self):
        self.table = dict()

    def __len__(self):
        return len(self.table)

    def add(self, docid, length):
        self.table[docid] = length

    def get_length(self, docid):
        if docid in self.table:
            return self.table[docid]
        else:
            raise LookupError('%s not found in table' % str(docid))

    def get_average_length(self):
        sum = 0
        for length in self.table.values():
            sum += length
        return float(sum) / float(len(self.table))


def main():
    # 1. 对问题进行加载
    qp = QueryParser(filename='./data/queries.txt')   # 加载问题
    qp.parse()
    queries = qp.get_queries()
    # print(queries)   # 得到问题的分词形式
    # [['portabl', 'oper', 'system'], ['code', 'optim', 'for', 'space', 'effici']]

    # 2. 对文章进行加载
    cp = CorpusParser(filename='./data/corpus.txt')    # 加载文章
    cp.parse()
    corpus = cp.get_corpus()
    # print(corpus)
    # {'1': [词1, 词2....], '2': [词1, 词2....], '3': [词1, 词2....]...}

    proc = QueryProcessor(queries, corpus)   # 对问题和文章进行简单的统计

    results = proc.run()   # {'1930': 9.646567300695821, '2246': 11.75268877838768, ...}
    # [{'1930': 9.646567300695821, '2246': 11.75268877838768}, {'2593': 6.5143785126884, '3127': 15.01712203629322} ...]
    # print(len(results))   # 7
    # print(len(results[0]))    # 871
    # print(len(results[1]))   # 1632

    qid = 0
    for result in results:
        sorted_x = sorted(result.items(), key=operator.itemgetter(1))
        sorted_x.reverse()
        index = 0
        for i in sorted_x[:10]:
            tmp = (qid, i[0], index, i[1])
            print('问题id:{}, 相关段落的id:{}, 最相关排序:{}, 相关得分:{}'.format(*tmp))
            index += 1
        exit()


        qid += 1


if __name__ == '__main__':
    main()
