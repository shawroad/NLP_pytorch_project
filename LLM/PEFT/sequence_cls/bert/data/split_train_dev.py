"""
@file   : split_train_dev.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2023-04-20
"""
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('./all_data.csv', sep='\t')
    print(df.shape)

    train_size = 0.95
    train_df = df.sample(frac=train_size, random_state=200)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(train_df.shape)
    print(val_df.shape)
    train_df.to_csv('./train.csv', index=False)
    val_df.to_csv('./val.csv', index=False)
