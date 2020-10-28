import random
import hashlib
import requests
from tqdm import tqdm


def baidu_translate(content, appid, secretKey, t_from='en', t_to='zh'):
    # 百度翻译方法
    # print(content)
    if len(content) > 4891:
        return '输入请不要超过4891个字符！'
    salt = str(random.randint(0, 50))
    # 申请网站 http://api.fanyi.baidu.com/api/trans
    # 这里写你自己申请的
    appid = appid
    # 这里写你自己申请的
    secretKey = secretKey
    sign = appid + content + salt + secretKey
    sign = hashlib.md5(sign.encode(encoding='UTF-8')).hexdigest()
    head = {'q': f'{content}',
            'from': t_from,
            'to': t_to,
            'appid': f'{appid}',
            'salt': f'{salt}',
            'sign': f'{sign}'}
    j = requests.get('http://api.fanyi.baidu.com/api/trans/vip/translate', head)
    res = j.json()['trans_result'][0]['dst']
    # print(res)
    return res


if __name__ == '__main__':
    with open('q_gov.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        result = []
        for line in tqdm(lines[:100]):
            line = line.strip()
            ori_question, ans = line.split('||')
            temp = baidu_translate(content=ori_question, appid='xxxx', secretKey='xxxxx', t_from='zh', t_to='en')    # 重点就这行和下面这行
            trans_question = baidu_translate(content=temp, appid='xxxxx', secretKey='xxxxx', t_from='en', t_to='zh')
            res = '||'.join([ori_question, trans_question, ans])
            result.append(res)
    with open('q_gov_aug.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(result))






