#!/usr/bin/env python3

import pandas as pd
import numpy as np
import nltk
import re
import argparse
from nltk.corpus import stopwords
from itertools import chain
nltk.download('stopwords', download_dir='/tmp')
nltk.data.path.append('/tmp')

def preprocess_tweets(tweet):
    """
    this function removes urls, stopwords and punctuations from the tweets
    :params: tweet = tweets
    
    """

    tweet = str(tweet)
    # Removing URL mentions
    tweet = ' '.join(re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet).split())
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    # Removing stopwords
    stop  = stopwords.words('english')
    tweet =' '.join([word for word in tweet.split() if word not in (stop)])
    # Removing punctuations
    tweet = tweet.replace('[^\w\s]','') 
    tweet = tweet.lower()
    
    return tweet
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Preprocess_tweets")
    parser.add_argument('--filename', type=str, help='add input file name')
    args = parser.parse_args()
    fname = args.filename
    tweets_df = pd.read_csv(fname)
    tweets_df['tweet_text'] = tweets_df.apply(lambda x: preprocess_tweets(x['tweet_text']), axis= 1)
    tweets_df.to_csv('preprocessed_' + fname, mode='a', index=False)