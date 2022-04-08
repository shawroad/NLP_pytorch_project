import json
import jieba
from collections import Counter
from tqdm import tqdm


if __name__ == '__main__':
    min_word_freq = 1
    samples = json.load(open('ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json', 'r', encoding='utf8'))
    temp = json.load(open('ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json', 'r', encoding='utf8'))
    samples.extend(temp)

    # Read image paths and captions for each image
    word_freq = Counter()

    for sample in tqdm(samples):
        caption = sample['caption']
        for c in caption:
            seg_list = jieba.cut(c, cut_all=True)
            # Update word frequency
            word_freq.update(seg_list)

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    print(len(word_map))
    print(words[:10])

    # Save word map to a JSON
    json.dump(word_map, open('WORDMAP.json', 'w', encoding='utf8'), ensure_ascii=False)


