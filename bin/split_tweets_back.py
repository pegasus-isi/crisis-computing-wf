#!/usr/bin/env python3

import pandas as pd
import numpy as np

PATH = "" 

if __name__ == "__main__":

    tweets_df = pd.read_csv('preprocessed_tweets.csv')
    num_tweets = len(tweets_df)
    tweets_df = tweets_df.sample(frac = 1) 
    
    trainset_size, valset_size, testset_size  = int(num_tweets * 0.7), int(num_tweets*0.15),int(num_tweets*0.15)
    train, validate, test = np.split(tweets_df, [trainset_size,(valset_size+trainset_size)])

    train.to_csv(PATH +"train_tweets.csv" ,index = False )
    validate.to_csv(PATH +"val_tweets.csv" ,index = False )
    test.to_csv(PATH + "test_tweets.csv" ,index = False )