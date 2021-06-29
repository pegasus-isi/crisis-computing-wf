#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.preprocessing import text, sequence
from keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools
import csv

max_len = 150
epochs = 3
max_features = 1000

DATA_PATH = ''
OUT_FILE_TRAIN = 'bilstm_train_output.csv'
OUT_FILE_TEST = 'bilstm_test_output.csv'

def prepare_data_train(train_df, test_df):
    
    
    X_train_id = train_df['tweet_id']
    X_train = train_df['tweet_text']
    Y_train = train_df['text_info']
    
    X_test_id = test_df['tweet_id']
    X_test = test_df['tweet_text']
    Y_test = test_df['text_info']

    tok = text.Tokenizer(num_words=max_features, lower=True)
    
    tok.fit_on_texts(list(X_train)+list(X_test))
    
    X_train = tok.texts_to_sequences(X_train)
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    
    X_test = tok.texts_to_sequences(X_test)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)
            
    return X_train, Y_train, X_train_id, X_test, Y_test, X_test_id


def evaluate(dtype, X_test_f, Y_test_f, model):
    
    accr = model.evaluate(X_test_f,Y_test_f) # NOT USED
    predictions = model.predict(X_test_f)
    p = 0.5
    output_class = np.where(predictions > 0.5, 1, 0)
    if dtype == 'test':
        cm =  confusion_matrix(Y_test_f, predictions > p)
        plot_confusion_matrix(cm, classes=["Informative","Noninforamtive"], normalize=True,
                          title='Normalized confusion matrix')

    return predictions, output_class, Y_test_f


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.3f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("bilstm_confusion_matrix.png")
                
        
def generate_csv(dtype, predicted_score, predicted_class, actual_class, tweet_id):
    
    if dtype == 'train':
        out_file = open(OUT_FILE_TRAIN, 'w+')
    else:
        out_file = open(OUT_FILE_TEST, 'w+') 
        
    csv_ob = csv.writer(out_file, delimiter=',')
    csv_ob.writerow(['tweet_id', 'predicted_score', 'predicted_class', 'actual_class'])
    tweet_id = list(tweet_id)
    actual_class = list(actual_class)
   
    for i in range(len(predicted_score)):
        csv_ob.writerow([tweet_id[i], predicted_score[i][0], predicted_class[i][0], actual_class[i]])
    
if __name__ == '__main__':
    
    train_df = pd.read_csv(DATA_PATH +'preprocessed_train_tweets.csv')
    valid_df = pd.read_csv(DATA_PATH +'preprocessed_val_tweets.csv')
    test_df  = pd.read_csv(DATA_PATH + 'preprocessed_test_tweets.csv')
    
    final_train_df = pd.concat([train_df, valid_df]) 
   
  
    X_train, Y_train, X_train_id, X_test, Y_test, X_test_id = prepare_data_train(final_train_df, test_df)

    model = keras.models.load_model('bilstm_final_model.h5')

    predictions, output_class, Y_test_f = evaluate('test', X_test, Y_test, model)
    generate_csv('test', predictions, output_class, Y_test_f, X_test_id)
    
    train_pred, train_output_class, Y_train_f = evaluate('train', X_train, Y_train, model)
    generate_csv('train', train_pred, train_output_class, Y_train_f, X_train_id)

    

