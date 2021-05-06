#!/usr/bin/env python3

import pandas as pd
import numpy as np

PATH = "" 
CACHE = {}

def check_duplicates(data):
    
    print("length of data before processing: ",len(data))
    for index, row in data.iterrows():
        if str(row['tweet_id']) not in CACHE.keys():
            CACHE[str(row['tweet_id'])] = 1
        else:
            data.drop(index, inplace=True)
    print("length of data after processing: ",len(data))
    
if __name__ == "__main__":

    tweets_df = pd.read_csv('all_tweets_df.csv')
    num_tweets = len(tweets_df)
 
    tweets_df = tweets_df.sample(frac = 1) 
    
    trainset_size, valset_size, testset_size  = int(num_tweets * 0.7), int(num_tweets*0.15),int(num_tweets*0.15)
 
    train, validate, test = np.split(tweets_df, [trainset_size,(valset_size+trainset_size)])
#     check_duplicates(train)
#     check_duplicates(validate)
#     check_duplicates(test)
    
    train.to_csv(PATH +"train_tweets.csv" ,index = False )
    validate.to_csv(PATH +"val_tweets.csv" ,index = False )
    test.to_csv(PATH + "test_tweets.csv" ,index = False )