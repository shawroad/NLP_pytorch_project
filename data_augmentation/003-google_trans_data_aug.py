# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/12 20:06
@Auth ： xiaolu
@File ：trans_data_aug.py
@IDE ：PyCharm
@Email：luxiaonlp@163.com
"""
import pandas as pd
from tqdm import tqdm as tqdm
import time
import execjs
import urllib.request


def open_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    req = urllib.request.Request(url=url, headers=headers)
    response = urllib.request.urlopen(req)
    data = response.read().decode('utf-8')
    return data


def translate(content, tk):
    """
    sl=en 代表源语言是英文，tl=zh-CN和hl=zh-CN代表目标语言是中文 。如需其他语言，自行更改。
    :param content:
    :param tk:
    :return:
    """
    if len(content) > 4891:
        print("翻译文本超过限制！")
        return
    content = urllib.parse.quote(content)
    # 中 -> 英
    # url = "http://translate.google.cn/translate_a/single?client=t" \
    #       "&sl=zh-CN&tl=en&hl=en&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca" \
    #       "&dt=rw&dt=rm&dt=ss&dt=t&ie=UTF-8&oe=UTF-8&clearbtn=1&otf=1&pc=1" \
    #       "&srcrom=0&ssel=0&tsel=0&kc=2&tk=%s&q=%s" % (tk, content)

    # 英 —> 中
    url = "http://translate.google.cn/translate_a/single?client=t" \
          "&sl=en&tl=zh-CN&hl=zh-CN&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca" \
          "&dt=rw&dt=rm&dt=ss&dt=t&ie=UTF-8&oe=UTF-8&clearbtn=1&otf=1&pc=1" \
          "&srcrom=0&ssel=0&tsel=0&kc=2&tk=%s&q=%s" % (tk, content)

    result = open_url(url)
    end = result.find("\",")
    end = 5
    #print(result)
    if end > 4:
        #print(result[4:end])
        return(result)


class Kaihua():
    def __init__(self):
        self.ctx = execjs.compile(""" 
        function TL(a) { 
        var k = ""; 
        var b = 406644; 
        var b1 = 3293161072; 

        var jd = "."; 
        var $b = "+-a^+6"; 
        var Zb = "+-3^+b+-f"; 

        for (var e = [], f = 0, g = 0; g < a.length; g++) { 
            var m = a.charCodeAt(g); 
            128 > m ? e[f++] = m : (2048 > m ? e[f++] = m >> 6 | 192 : (55296 == (m & 64512) && g + 1 < a.length && 56320 == (a.charCodeAt(g + 1) & 64512) ? (m = 65536 + ((m & 1023) << 10) + (a.charCodeAt(++g) & 1023), 
            e[f++] = m >> 18 | 240, 
            e[f++] = m >> 12 & 63 | 128) : e[f++] = m >> 12 | 224, 
            e[f++] = m >> 6 & 63 | 128), 
            e[f++] = m & 63 | 128) 
        } 
        a = b; 
        for (f = 0; f < e.length; f++) a += e[f], 
        a = RL(a, $b); 
        a = RL(a, Zb); 
        a ^= b1 || 0; 
        0 > a && (a = (a & 2147483647) + 2147483648); 
        a %= 1E6; 
        return a.toString() + jd + (a ^ b) 
    }; 

    function RL(a, b) { 
        var t = "a"; 
        var Yb = "+"; 
        for (var c = 0; c < b.length - 2; c += 3) { 
            var d = b.charAt(c + 2), 
            d = d >= t ? d.charCodeAt(0) - 87 : Number(d), 
            d = b.charAt(c + 1) == Yb ? a >>> d: a << d; 
            a = b.charAt(c) == Yb ? a + d & 4294967295 : a ^ d 
        } 
        return a 
    } 
    """)
    def getTk(self, text):
        return self.ctx.call("TL", text)


if __name__ == '__main__':
    js = Kaihua()
    success = 0
    # a_list = ['我的花呗预期了','我的花呗逾期了','苹果7什么时候上市','苹果5什么时候上市','一定要去4S店才能更换后视镜吗？',
    #           '汽车在升级程序后就无法读取系统这是什么个情况','我都等三个月了','那你们系统是根据什么把我的花呗停掉的',
    #           '为什么还开不了花呗借呗什么的','北京到广州顺丰快递要多久','顺丰快递广州到北京要多久','尽情花呗是什么情况',
    #           '尽情花呗如何开','不小心开通花呗','我想开通花呗？不知道怎么操作','给泰迪起什么名字好听','泰迪犬取什么名字好听，是MM！']
    a_list = ['what is your name?', 'what are you doing?']
    a_trans_list = []   # a_trans_list 保存翻译后的句子列表。
    """
    总体流程
    每100句话用'\n'作为分隔符组成一个文档放入谷歌进行翻译，翻译回来后解析回原来的100句话。
    单句单句输入太费时间。如此设计10w短文本的翻译时间大约是20min。
    注意，100句话总字数不能超过5000字，这是谷歌翻译的限制。因此可根据文本长度自由调节这个combined_length值.
    """
    combined_length = 100
    for i in tqdm(range(len(a_list)//combined_length+1)):
        content = '\n'.join(a_list[i*combined_length:(i+1)*combined_length])
        if content == '':
            continue
        get_trans = True
        while get_trans:
            try:
                tk = js.getTk(content)
                result = translate(content, tk)
                result = result.replace('null', 'None')
                result = result.replace('true', 'True')
                result = result.replace('false', 'False')
                result = eval(result)
                trans_result = ''
                for item in result[0][0:-1]:
                    trans_result += item[0]
                train_result = trans_result.split('\n')
                print(len(train_result), len(content.split('\n')))
                if len(train_result) != len(content.split('\n')):
                    print('wrong')
                    raise Exception
                else:
                    pass
                a_trans_list.extend(train_result)
                success += 1
                get_trans = False
                time.sleep(0.1)
            except:
                pass
        if success % 10 == 0:
            print(success)
            print(a_trans_list[-10:])
        time.sleep(0.1)

    assert len(a_list) == len(a_trans_list)
    print('翻译顺利结束！')
    print(a_trans_list)

