import json


if __name__ == '__main__':
    all_corpus = []
    with open('train.json', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            all_corpus.append(len(line['text']))
    print(max(all_corpus))
    print(min(all_corpus))
    print(len(all_corpus))
    
    
