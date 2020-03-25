"""

@file  : extract_paragraph.py

@author: xiaolu

@time  : 2020-03-02

"""
import sys
import json
import copy
from tqdm import tqdm
from utils import metric_max_over_ground_truths, f1_score


def compute_paragraph_score(sample):
    '''
    对于每段，计算和问题的f1-score
    :param sample:
    :return:
    '''
    scores = []
    question = sample['segmented_question']   # 取出问题的分词形式(还是中文 不是id)

    for doc in sample['documents']:
        doc['segmented_paragraphs_scores'] = []  # 给每篇文章加个域 段落匹配得分
        for p_idx, para_tokens in enumerate(doc['segmented_paragraphs']):   # 此处遍历的是一篇文章的每段话(分词形式)
            if len(question) > 0:
                related_score = metric_max_over_ground_truths(f1_score, para_tokens, [question])
            else:
                related_score = 0.0

            doc['segmented_paragraphs_scores'].append(related_score)  # 每段话与问题的得分
            scores.append(related_score)   # 获取每段文字与问题的相似得分


def dup_remove(doc):
    '''
    For each document, remove the duplicated paragraphs
    :param doc:
    :return:
    '''
    paragraphs_his = {}
    del_ids = []
    para_id = None

    if 'most_related_para' in doc:
        # 当前文章最关键的段落
        para_id = doc['most_related_para']   # 当前文章中与问题最相关的段落

    # 下面的for循环就是删除文章中的重复段落,还有就是这样的话　有可能把para_id(最相关段落)搞乱　可能会涉及到重置
    doc['paragraphs_length'] = []   # 段落长度
    for p_idx, (segmented_paragraph, paragraph_score) in enumerate(zip(doc['segmented_paragraphs'], doc['segmented_paragraphs_scores'])):
        doc['paragraphs_length'].append(len(segmented_paragraph))   # 每个文本加一个域　距离当前文章中每段话的长度
        paragraph = ''.join(segmented_paragraph)   # 当前这一段话连起来
        if paragraph in paragraphs_his:
            # 这里主要删除的是重复的话
            del_ids.append(p_idx)
            if p_idx == para_id:
                para_id = paragraphs_his[paragraph]
            continue
        paragraphs_his[paragraph] = p_idx   # 如果当前段落没有在paragraphs_his中　我们将其标号　

    # delete 上面的for 是找到重复的段落　接下来删除
    prev_del_num = 0
    del_num = 0
    for p_idx in del_ids:
        if p_idx < para_id:
            prev_del_num += 1  # 这里的if也是防止para_id(最相关段落)搞乱了, 所以要统计para_id之前删了几段话　等会可以重置

        # 删除重复的段落
        del doc['segmented_paragraphs'][p_idx - del_num]
        del doc['segmented_paragraphs_scores'][p_idx - del_num]
        del doc['paragraphs_length'][p_idx - del_num]
        del_num += 1

    if len(del_ids) != 0:
        if 'most_related_para' in doc:
            doc['most_related_para'] = para_id - prev_del_num   # 重置最相关段落的标号
        doc['paragraphs'] = []
        for segmented_para in doc['segmented_paragraphs']:   # 把剩余的段落加进来
            paragraph = ''.join(segmented_para)
            doc['paragraphs'].append(paragraph)
        return True
    else:
        return False


def paragraph_selection(sample, mode):
    '''
    For each document, select paragraphs that includes as much information as possible
    :param sample: a sample in the dataset
    :param mode: string of ('train', 'dev', 'test'), indicate the type of dataset to process.
    :return:
    '''
    scores = []
    MAX_P_LEN = 510   # 定长的文本长度为510

    splitter = '。'

    # 选取最相关的五段  topN of related paragraph to choose
    topN = 5
    doc_id = None

    # answer_docs 答案出自的那篇文章
    if 'answer_docs' in sample and len(sample['answer_docs']) > 0:
        doc_id = sample['answer_docs'][0]   # 取出答案所在的那篇文章的id
        if doc_id >= len(sample['documents']):
            # 文章id 比我们文章总数还多  显然是扯淡
            return

    for d_idx, doc in enumerate(sample['documents']):
        # 看每篇文章与问题的匹配得分
        if 'segmented_paragraphs_scores' not in doc:
            # 看每段与答案的得分 若没有 扔掉当前文本
            continue

        # 删除文章的重复段落　可能会搞乱最相关段落的id　这里会涉及到更改最相关段落的id
        status = dup_remove(doc)
        # print(status)   # 返回True代表有相同段落已经被删除　返回False表示没有删除段落

        segmented_title = doc['segmented_title']
        title_len = len(segmented_title)
        para_id = None
        if doc_id is not None:
            para_id = sample['documents'][doc_id]['most_related_para']  # 取出答案出自那篇文章的最相关段落

        # 问题 ＋ 文章 的长度
        total_len = title_len + sum(doc['paragraphs_length'])

        # add splitter
        para_num = len(doc['segmented_paragraphs'])   # 总共有几段话
        total_len += para_num  # 这里之所以要加段落的个数　是考虑到加句号　连接句子

        if total_len <= MAX_P_LEN:
            incre_len = title_len    # 标题的长度
            total_segmented_content = copy.deepcopy(segmented_title)    # 拷贝标题的分词形式

            # 这里强调一下doc_id是标注的答案出现在哪篇文章中
            for p_idx, segmented_para in enumerate(doc["segmented_paragraphs"]):
                if doc_id == d_idx and para_id > p_idx:
                    # 你的id是答案出自的那篇文章　并且你的id在最相关段落的id之前
                    incre_len += len([splitter] + segmented_para)

                if doc_id == d_idx and para_id == p_idx:
                    # 你的id是答案出自的那篇文章 并且你的id是最相关段落的id
                    incre_len += 1
                total_segmented_content += [splitter] + segmented_para

            # 更改答案的起始和结束标志
            if doc_id == d_idx:
                answer_start = incre_len + sample['answer_spans'][0][0]
                answer_end = incre_len + sample['answer_spans'][0][1]
                sample['answer_spans'][0][0] = answer_start
                sample['answer_spans'][0][1] = answer_end
            doc["segmented_paragraphs"] = [total_segmented_content]
            doc["segmented_paragraphs_scores"] = [1.0]
            doc['paragraphs_length'] = [total_len]
            doc['paragraphs'] = [''.join(total_segmented_content)]
            doc['most_related_para'] = 0
            continue

        # find topN paragraph id
        # 把当前每段话的分词形式, 段得分, 段的长度 组成元祖  还有p_idx 放进列表中
        para_infos = []
        for p_idx, (para_tokens, para_scores) in enumerate(zip(doc['segmented_paragraphs'], doc['segmented_paragraphs_scores'])):
            para_infos.append((para_tokens, para_scores, len(para_tokens), p_idx))

        # 取出前面的几篇文章
        topN_idx = []

        for para_info in para_infos[:topN]:
            topN_idx.append(para_info[-1])
            scores.append((para_info[1], para_info[2]))

        final_idx = []
        total_len = title_len
        if doc_id == d_idx:
            if mode == "train":
                final_idx.append(para_id)
                total_len = title_len + 1 + doc['paragraphs_length'][para_id]

        for id in topN_idx:
            if total_len > MAX_P_LEN:
                break
            if doc_id == d_idx and id == para_id and mode == "train":
                continue
            total_len += 1 + doc['paragraphs_length'][id]
            final_idx.append(id)

        total_segmented_content = copy.deepcopy(segmented_title)
        final_idx.sort()
        incre_len = title_len
        for id in final_idx:
            if doc_id == d_idx and id < para_id:
                incre_len += 1 + doc['paragraphs_length'][id]
            if doc_id == d_idx and id == para_id:
                incre_len += 1
            total_segmented_content += [splitter] + doc['segmented_paragraphs'][id]
        if doc_id == d_idx:
            answer_start = incre_len + sample['answer_spans'][0][0]
            answer_end = incre_len + sample['answer_spans'][0][1]
            sample['answer_spans'][0][0] = answer_start
            sample['answer_spans'][0][1] = answer_end
        doc["segmented_paragraphs"] = [total_segmented_content]
        doc["segmented_paragraphs_scores"] = [1.0]
        doc['paragraphs_length'] = [total_len]
        doc['paragraphs'] = [''.join(total_segmented_content)]
        doc['most_related_para'] = 0


if __name__ == '__main__':
    # 抽取训练数据集
    path = './trainset/search.train.json'
    mode = 'train'
    total = ''
    with open(path) as f:
        for line in tqdm(f.readlines()):
            sample = json.loads(line)  # 加载一条数据集
            # 计算每篇文中的每段话与问题的匹配得分
            compute_paragraph_score(sample)
            paragraph_selection(sample, mode)
            s = json.dumps(sample, ensure_ascii=False)
            total += s + '\n'
    with open('./extract/train/search.train.json', 'w') as f:
        f.write(total)
    print("训练数据集中的search语句搞完了")

    path = './trainset/zhidao.train.json'
    mode = 'train'
    total = ''
    with open(path) as f:
        for line in tqdm(f.readlines()):
            sample = json.loads(line)

            compute_paragraph_score(sample)

            paragraph_selection(sample, mode)
            s = json.dumps(sample, ensure_ascii=False)
            total += s + '\n'
    with open('./extract/train/zhidao.train.json', 'w') as f:
        f.write(total)
    print("训练数据集中的zhidao语句搞完了")

    # dev
    path = './devset/search.dev.json'
    mode = 'dev'
    total = ''
    with open(path) as f:
        for line in tqdm(f.readlines()):
            sample = json.loads(line)

            compute_paragraph_score(sample)

            paragraph_selection(sample, mode)
            s = json.dumps(sample, ensure_ascii=False)
            total += s + '\n'
    with open('./extract/dev/search.dev.json', 'w') as f:
        f.write(total)
    print("验证数据集中的seach语句搞完了")

    path = './devset/zhidao.dev.json'
    mode = 'dev'
    total = ''
    with open(path) as f:
        for line in tqdm(f.readlines()):
            sample = json.loads(line)

            compute_paragraph_score(sample)

            paragraph_selection(sample, mode)
            s = json.dumps(sample, ensure_ascii=False)
            total += s + '\n'
    with open('./extract/dev/zhidao.dev.json', 'w') as f:
        f.write(total)
    print("验证数据集中的zhidao语句搞完了")

    # test
    path = './testset/search.test1.json'
    mode = 'test'
    total = ''
    with open(path) as f:
        for line in tqdm(f.readlines()):
            sample = json.loads(line)

            compute_paragraph_score(sample)

            paragraph_selection(sample, mode)
            s = json.dumps(sample, ensure_ascii=False)
            total += s + '\n'
    with open('./extract/test/search.test1.json', 'w') as f:
        f.write(total)
    print("测试数据集中的search语句搞完了")

    path = './testset/zhidao.test1.json'
    mode = 'test'
    total = ''
    with open(path) as f:
        for line in tqdm(f.readlines()):
            sample = json.loads(line)

            compute_paragraph_score(sample)

            paragraph_selection(sample, mode)
            s = json.dumps(sample, ensure_ascii=False)
            total += s + '\n'
    with open('./extract/test/zhidao.test1.json', 'w') as f:
        f.write(total)
    print("测试数据集中的zhidao语句搞完了")


'''
加一段中文注释:
  1. 首先算了一下各篇文章中每段话与问题的相关性, 具体方法: 统计共现词 然后看共现词在那段话中的比重 比重大 说明重要
  2. 然后删除每篇文章中的重复段落. 这里要考虑每篇文章中最相关段落有可能会打乱 所以 要小心点
  3. 因为这是我们的标注数据,所以有答案. 我们根据标志的最相关文章  把其余文章删除掉。 同时处理文章，把最相关段落以后的段落删除掉 只要前面部分
     然后将每段话用句号连接起来。
  4. 


'''