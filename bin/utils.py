#!/usr/bin/env python3

import glob
import os
import random
import pandas as pd
import numpy as np
import csv
import re

REL = os.getcwd()
INFORMATIVE_IMAGES_1 = REL + '/dataset_temp/Training_data/Informative'
INFORMATIVE_IMAGES_2 = REL + '/dataset_temp/Testing_data/Informative'
NON_INFORMATIVE_IMAGES_1 = REL + '/dataset_temp/Testing_data/Non-Informative'
NON_INFORMATIVE_IMAGES_2 = REL + '/dataset_temp/Training_data/Non-Informative'

INFORMATIVE_TWEETS = REL + '/dataset_temp/tweets_csv/INFORMATIVE_TWEETS/'
NON_INFORMATIVE_TWEETS = REL + '/dataset_temp/tweets_csv/NONINFORMATIVE_TWEETS/'

FINAL_TWEETS = REL + '/dataset_temp/final_tweets/'

def add_prefix(data, prefix):
    """
    adds prefix train/test/val to the images
    
    """
    dataset = []
    for imgs in data:
        path = '/'.join(imgs.split('/')[:-1])
        name = imgs.split('/')[-1]
        new_name = os.path.join(path, (prefix + name))
        os.rename(imgs, new_name)
        dataset.append(new_name)
        
    return dataset
    
def get_specific_images(text_df, image_id_name_dict):
    
    images = []
    for index, row in text_df.iterrows():
        tweet_id = str(row['tweet_id'])
        images.extend(image_id_name_dict[tweet_id])
    
    return images

def get_image_splits(train_tweets, val_tweets, test_tweets, image_id_name_dict):
    """
    splits the dataset into train, validation and testing with ratio 80-10-10.
    
    """
    train_tweet_df = pd.read_csv(train_tweets)
    val_tweet_df = pd.read_csv(val_tweets)
    test_tweet_df = pd.read_csv(test_tweets)
    
    train = get_specific_images(train_tweet_df, image_id_name_dict)
    val = get_specific_images(val_tweet_df, image_id_name_dict)
    test = get_specific_images(test_tweet_df, image_id_name_dict)
    
    train = add_prefix(train, 'train_')
    val = add_prefix(val, 'val_')
    test = add_prefix(test, 'test_')
    
    dataset = train + val + test
    
    return dataset


def get_images():
    """
    returns informative and non-informative images with .png and .jpg extension 
    
    """

    informative = glob.glob(INFORMATIVE_IMAGES_1+'/*.png') + glob.glob(INFORMATIVE_IMAGES_2+'/*.png') + glob.glob(INFORMATIVE_IMAGES_1+'/*.jpg') + glob.glob(INFORMATIVE_IMAGES_2+'/*.jpg')

    non_informative = glob.glob(NON_INFORMATIVE_IMAGES_1+'/*.png') + glob.glob(NON_INFORMATIVE_IMAGES_2+'/*.png') + glob.glob(NON_INFORMATIVE_IMAGES_1+'/*.jpg') + glob.glob(NON_INFORMATIVE_IMAGES_2+'/*.jpg')
    
    return informative, non_informative
    
    
def read_data(files):
    """
    combines csv files of tweets as a dataframe and returns a final single dataframe object
    """
    dfs = []
    for file in files:
        df = pd.read_csv(file, error_bad_lines=False)
        dfs.append(df)
    final_df = pd.concat(dfs)
    return final_df  


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
        
def get_ids(images):
   
    image_ids = {}
    for image in images:
        image_name = image.split('/')[-1]
        img_id = str(re.findall(r'[0-9]+', image_name)[0])
        
        if img_id in image_ids.keys():
            image_ids[img_id].append(image)
        else:
            image_ids[img_id] = [image]
            
    return list(image_ids.keys()), image_ids

def get_tweets(image_ids):
    """
    returns informative and non-informative tweets 
    """
    informative_files = glob.glob(INFORMATIVE_TWEETS + '*.csv')
    noninformative_files = glob.glob(NON_INFORMATIVE_TWEETS + '*.csv')
    
    info_df = read_data(informative_files)
    noninfo_df = read_data(noninformative_files)
    
    info_df["text_info"] = 1
    noninfo_df["text_info"] = 0
    
    final_df = pd.concat([info_df, noninfo_df])
    
    if not os.path.isdir(FINAL_TWEETS): 
        os.mkdir(FINAL_TWEETS)
    
    filename = "all_tweets_df.csv"
    file = open(FINAL_TWEETS + filename, 'w+')
    
    refactor_tweets(file, final_df, image_ids)
    
    PATH_FINAL_TWEETS = FINAL_TWEETS + 'all_tweets_df.csv'
    
    return PATH_FINAL_TWEETS


def split_tweets(path):
    
    tweets_df = pd.read_csv(path)
    num_tweets = len(tweets_df)
    tweets_df = tweets_df.sample(frac = 1) 
    
    trainset_size, valset_size, testset_size  = int(num_tweets * 0.7), int(num_tweets*0.15), int(num_tweets*0.15)
 
    train, validate, test = np.split(tweets_df, [trainset_size-1,(valset_size+trainset_size-1)])
    
    path_train = FINAL_TWEETS + "train_tweets.csv"
    path_val = FINAL_TWEETS + "val_tweets.csv"
    path_test = FINAL_TWEETS + "test_tweets.csv"
    
    train.to_csv(path_train ,index = False )
    validate.to_csv(path_val ,index = False )
    test.to_csv(path_test ,index = False )
    
    return path_train, path_val, path_test
    
def add_labels(images, label):
    """
    attaches label _1 if class in informative or _0 if class in non-informative
    :params: images and their class
    
    """
    labelled_images = []
    if label == 'informative':
        prefix = '_1'
    else:
        prefix = '_0'

    for image in images:
        name = image.split('/')[-1]
        path = '/'.join(image.split('/')[:-1])
        new_name = os.path.join(path, (name[:-4]+prefix+name[-4:]))
        os.rename(image, new_name)
        labelled_images.append(new_name)
    return labelled_images

