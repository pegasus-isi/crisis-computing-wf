#!/usr/bin/env python3

import glob
import pandas as pd
import numpy as np
import re
import csv

PATH = ''
CACHE = {}

def create_csv(file, data_id, tweets_df):
    
    csv_ob = csv.writer(file, delimiter=',')
    csv_ob.writerow(['tweet_id', 'tweet_text', 'text_info'])

    for ids in data_id:       
        if ids in CACHE.keys():
            continue
        row = tweets_df.loc[tweets_df['tweet_id'] == ids]
        csv_ob.writerow([row['tweet_id'].values[0], row['tweet_text'].values[0], row['text_info'].values[0]])
        CACHE[ids] = 1
    

if __name__ == "__main__":
    
    train_images = glob.glob(PATH + 'train_*.png') + glob.glob(PATH + 'train_*.jpg')
    val_images = glob.glob(PATH + 'val_*.png') + glob.glob(PATH + 'val_*.jpg')
    test_images = glob.glob(PATH + 'test_*.png') + glob.glob(PATH + 'test_*.jpg')

    train_ids = [ re.findall(r'[0-9]+', x.split('/')[-1])[0] for x in train_images]
    val_ids = [ re.findall(r'[0-9]+', x.split('/')[-1])[0] for x in val_images]
    test_ids = [ re.findall(r'[0-9]+', x.split('/')[-1])[0] for x in test_images]

    tweets_df = pd.read_csv('preprocessed_tweets.csv', index_col=False)
    tweets_df['tweet_id'] = tweets_df.tweet_id.astype(str)

    test_df = open(PATH + "test_tweets.csv", 'w+')
    train_df = open(PATH + "train_tweets.csv", 'w+')
    val_df = open(PATH + "val_tweets.csv", 'w+')

    create_csv(test_df, test_ids, tweets_df)
    create_csv(train_df, train_ids, tweets_df)
    create_csv(val_df, val_ids, tweets_df)