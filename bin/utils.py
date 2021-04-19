#!/usr/bin/env python3

import glob
import os
import random
import pandas as pd
import numpy as np

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
#         os.rename(imgs, new_name)
        dataset.append(new_name)
        
    return dataset
    
def get_data_splits():
    """
    splits the dataset into train, validation and testing with ratio 80-10-10.
    
    """
    dataset = {}
    informative, non_informative = get_images()
    data = informative + non_informative
    random.shuffle(data)
    
    train, val, test = np.split(data, [int(len(data)*0.8), int(len(data)*0.9)])
    
#     train = add_prefix(train, 'train_')
#     val = add_prefix(val, 'val_')
#     test = add_prefix(test, 'test_')
    
    dataset['train'], dataset['val'], dataset['test'] = train, val, test
    
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


def get_tweets():
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
    
   
    os.mkdir(FINAL_TWEETS)
    filename = "all_tweets_df.csv" 
    final_df.to_csv(FINAL_TWEETS + filename, index = False )
    
    PATH_FINAL_TWEETS = FINAL_TWEETS + 'all_tweets_df.csv'
    
    return PATH_FINAL_TWEETS, filename

    
def add_labels(images, label):
    """
    attaches label _1 if class in informative or _0 if class in non-informative
    :params: images and their class
    
    """
    if label == 'informative':
        prefix = '_1'
    else:
        prefix = '_0'

    for image in images:
        name = image.split('/')[-1]
        path = '/'.join(image.split('/')[:-1])
        new_name = os.path.join(path, (name[:-4]+prefix+name[-4:]))
#         os.rename(image, new_name)
    
    return



