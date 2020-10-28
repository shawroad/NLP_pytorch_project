#!/usr/bin/python
# pip install nlpcda -i https://pypi.douban.com/simple/
from nlpcda import Randomword
from nlpcda import Similarword
from nlpcda import Homophone
from nlpcda import RandomDeleteChar
from nlpcda import Ner
from nlpcda import CharPositionExchange
from nlpcda import baidu_translate
from nlpcda import EquivalentChar


def test_EquivalentChar(test_str, create_num=2, change_rate=0.5):
    '''
    同义字替换
    :param test_str: 原始串
    :param create_num: 返回多少个串
    :param change_rate: 每次改变的概率
    :return:
    '''
    s = EquivalentChar(create_num=create_num, change_rate=change_rate)
    return s.replace(test_str)


def test_Randomword(test_str, create_num=2, change_rate=0.2):
    '''
    等价实体替换  这里是extdata/company.txt ，随机公司实体替换
    :param test_str: 替换文本
    :param create_num: 增强为多少个
    :param change_rate: 文本变化率/最大多少比例会被改变
    :return:
    '''
    smw = Randomword(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)


def test_Similarword(test_str, create_num=2, change_rate=0.2):
    '''
    随机同义词替换
    :param test_str: 替换文本
    :param create_num: 增强为多少个
    :param change_rate: 文本变化率/最大多少比例会被改变
    :return:
    '''
    smw = Similarword(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)


def test_Homophone(test_str, create_num=2, change_rate=0.2):
    '''
    随机【同意/同音字】替换
    :param test_str: 替换文本
    :param create_num: 增强为多少个
    :param change_rate: 文本变化率/最大多少比例会被改变
    :return:
    '''
    hoe = Homophone(create_num=create_num, change_rate=change_rate)
    return hoe.replace(test_str)


def test_RandomDeleteChar(test_str, create_num=2, change_rate=0.1):
    # 随机删除字符
    smw = RandomDeleteChar(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)


def test_CharPositionExchange(test_str, create_num=2, change_rate=0.5):
    # 位置交换
    smw = CharPositionExchange(create_num=create_num, change_rate=change_rate)
    return smw.replace(test_str)


def test_baidu_translate(test_str):
    # 申请你的 appid、secretKey
    # 两遍洗数据法（回来的中文一般和原来不一样，要是一样，就不要了，靠运气？）
    temp = baidu_translate(content=test_str, appid='XXXX', secretKey='XXXX', t_from='zh', t_to='en')
    res = baidu_translate(content=temp, appid='XXXX', secretKey='XXXX', t_from='en', t_to='zh')
    return res


if __name__ == '__main__':
    # 等价字替换
    ts = '''这是个实体：58同城；金山软件今天是2020年3月8日11:40，天气晴朗，天气很不错，空气很好，不差；这个nlpcad包，用于方便一键数据增强，可有效增强NLP模型的泛化性能、减少波动、抵抗对抗攻击'''
    rs1 = test_EquivalentChar(ts)
    print('*'*10 + '等价字替换' + '*'*10)
    for _ in rs1:
        print(_)

    # 等价实体替换
    rs2 = test_Randomword(ts)
    print('*' * 10 + '等价实体替换' + '*' * 10)
    for _ in rs2:
        print(_)

    # 同义词替换
    rs3 = test_Similarword(ts)
    print('*' * 10 + '同义词替换' + '*' * 10)
    for _ in rs3:
        print(_)

    # 同音字替换
    rs4 = test_Homophone(ts)
    print('*' * 10 + '同音字替换' + '*' * 10)
    for _ in rs4:
        print(_)

    # 随机删除字符
    rs5 = test_RandomDeleteChar(ts)
    print('*' * 10 + '随机删除字符' + '*' * 10)
    for _ in rs5:
        print(_)

    # 位置交换
    rs6 = test_CharPositionExchange(ts)
    print('*' * 10 + '位置交换' + '*' * 10)
    for _ in rs6:
        print(_)

    # 翻译
    print('*' * 10 + '翻译增广' + '*' * 10)
    print(ts)
    print(test_baidu_translate(ts))
