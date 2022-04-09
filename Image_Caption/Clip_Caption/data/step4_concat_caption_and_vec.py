import pickle
import json
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    path1 = './corpus/ai_challenge_train_caption.csv'
    path2 = './corpus/ai_challenge_val_caption.csv'
    path3 = './corpus/filckr_caption.csv'
        
    caption_path = [path1, path2, path3]
    df = pd.DataFrame()
    for p in caption_path:
        temp_df = pd.read_csv(p)
        df = pd.concat([df, temp_df])
    
    
    print(df.shape)   # (1388915, 2)
    
    
    # print(df.head())
    '''
    image_id                            caption
    8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg                  两个衣着休闲的人在平整的道路上交谈
    '''
    
    e_path1 = './corpus/ai_train_image_id2embed.pkl'
    embed = pickle.load(open(e_path1, 'rb'))
    
    # ai_val_image_id2embed.pkl
    e_path2 = './corpus/ai_val_image_id2embed.pkl'
    embed2 = pickle.load(open(e_path2, 'rb'))
    
    e_path3 = './corpus/flickr30k_image_id2embed.pkl'
    embed3 = pickle.load(open(e_path3, 'rb'))
    
    embed.update(embed2)
    embed.update(embed3)
    
    pickle.dump(embed, open('./corpus/all_imageid2embed.pkl', 'wb'))
    
    all_key = set(list(embed.keys()))
    
    final_image_id, final_caption = [], []
    for image_id, caption in tqdm(zip(df['image_id'].tolist(), df['caption'].tolist())):
        if len(str(caption)) < 4:
            continue
        if len(set([image_id]) & all_key) >= 1:
            final_image_id.append(image_id)
            final_caption.append(caption)
    
    df = pd.DataFrame({'image_id': final_image_id, 'caption': final_caption})
    df.to_csv('./corpus/all_caption.csv', index=False)
    print(df.shape)
