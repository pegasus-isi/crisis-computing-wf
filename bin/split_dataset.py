#!/usr/bin/env python3

import pandas as pd
import numpy as np
import glob
from utils import add_prefix
import csv

PATH = "" 
CACHE = {}

def get_images(data_ids):
    """
    returns images and the unique ids based on textual data split
    """
    
    final_ids = []
    images = []
    for value in data_ids:
        value = str(value)
        imgs = glob.glob('*'+value+'*.png') + glob.glob('*'+value+'*.jpg')
        if imgs == [] or imgs == None:
            continue
        images.extend(imgs)
        final_ids.append(value)
        
    return images, final_ids
   
def refactor_tweets(file, tweets_df, tweet_ids):
    """
    creates new text csv file with data present in image pipeline
    """
    csv_ob = csv.writer(file, delimiter=',')
    csv_ob.writerow(['tweet_id', 'tweet_text', 'text_info'])
    tweets_df['tweet_id'] = tweets_df.tweet_id.astype(str)
    
    for ids in tweet_ids:
        row = tweets_df.loc[tweets_df['tweet_id'] == ids]
        csv_ob.writerow([row['tweet_id'].values[0], row['tweet_text'].values[0], row['text_info'].values[0]])
    
            
if __name__ == "__main__":

    train_tweets = pd.read_csv('train_tweets.csv')
    val_tweets = pd.read_csv('val_tweets.csv')
    test_tweets = pd.read_csv('test_tweets.csv')
    
    train_ids = train_tweets['tweet_id'].values
    val_ids = val_tweets['tweet_id'].values
    test_ids = test_tweets['tweet_id'].values
    
    train_images, final_train_ids = get_images(train_ids)
    val_images, final_val_ids = get_images(val_ids)
    test_images, final_test_ids = get_images(test_ids)
    
    train = add_prefix(train_images, 'train_')
    val = add_prefix(val_images, 'val_')
    test = add_prefix(test_images, 'test_')
    
    test_file = open("new_test_tweets.csv", 'w+')
    train_file = open("new_train_tweets.csv", 'w+')
    val_file = open("new_val_tweets.csv", 'w+')
    
    refactor_tweets(train_file, train_tweets, final_train_ids)
    refactor_tweets(test_file, test_tweets, final_test_ids)
    refactor_tweets(val_file, val_tweets, final_val_ids)
    
    