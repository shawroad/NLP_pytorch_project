import json
from tqdm import tqdm
import pandas as pd



if __name__ == '__main__':
    path = './corpus/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json'
    data = json.load(open(path, 'r', encoding='utf8'))
    
    image_id_list, caption_list = [], []
    for item in tqdm(data):
        caption_list.extend(item['caption'])
        image_id_list.extend([item['image_id']] * len(item['caption']))
    
    df = pd.DataFrame({'image_id': image_id_list, 'caption': caption_list})
    df.to_csv('./corpus/ai_challenge_val_caption.csv', index=False)
    print(df.shape)
    
    

        
    
    
