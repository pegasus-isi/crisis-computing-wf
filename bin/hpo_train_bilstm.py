#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import optuna
import joblib
import os
import matplotlib.pyplot as plt
from keras.layers import Dense,Input, LSTM, Bidirectional, Activation, Conv1D, GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import itertools

target_names = ["Informative","Noninforamtive"]

max_len = 150
embed_size = 200
EMBEDDING_FILE = 'glove.twitter.27B.200d.txt'
class_weight = {0: 1.48, 1: 0.75}
epochs = 10


METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')
]


### ------------------------- DATA -------------------------------------

DATA_PATH = ''


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
def BiLSTM(LR, dropout_val, embedding_matrix, max_features):
    
    sequence_input = Input(shape=(max_len, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = False)(sequence_input)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool]) 
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_val)(x)
    preds = Dense(1, activation="sigmoid")(x)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=LR),metrics=METRICS)
    
    return model    

 # unbalanced dataset - get weights
def get_weights(train_val_df):
    
    neg, pos = np.bincount(train_val_df)
    total = neg + pos
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    return class_weight


def prepare_data_train(train_df, valid_df, test_df, embeddings_index, max_features):

    X_train = train_df['tweet_text']
    Y_train = train_df['text_info']

    X_valid = valid_df['tweet_text']
    Y_valid = valid_df['text_info']

    X_test = test_df['tweet_text']
    Y_test = test_df['text_info']

    tok = text.Tokenizer(num_words=max_features, lower=True)
    
    tok.fit_on_texts(list(X_train)+list(X_test)+list(X_valid))

    X_train = tok.texts_to_sequences(X_train)
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)

    X_valid = tok.texts_to_sequences(X_valid)
    X_valid = sequence.pad_sequences(X_valid, maxlen=max_len)

    X_test = tok.texts_to_sequences(X_test)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)

    word_index = tok.word_index
    
    #prepare embedding matrix
    
    
    num_words = min(max_features, len(word_index) + 1)
   
    embedding_matrix = np.zeros((num_words, embed_size))
    
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix, X_train, Y_train, X_test, Y_test, X_valid, Y_valid, num_words



def get_embeddings():
    embeddings_index = {}

    with open(EMBEDDING_FILE,encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def objective(trial):
    
    print("Performing trial {}".format(trial.number))
    global max_features
    
    train_df = pd.read_csv(DATA_PATH +'preprocessed_train_tweets.csv')
    valid_df = pd.read_csv(DATA_PATH +'preprocessed_val_tweets.csv')
    test_df  = pd.read_csv(DATA_PATH + 'preprocessed_test_tweets.csv')
    
    LR = trial.suggest_categorical("LR", [1e-3, 1e-4, 1e-5])
    dropout_val = trial.suggest_categorical("dropout_val", [0.1, 0.2, 0.4])
    
    embeddings_index = get_embeddings()
    max_features = 1000
    embedding_matrix, X_train, Y_train, X_test, Y_test, X_valid, Y_valid, embed_size = prepare_data_train(train_df, valid_df, test_df, embeddings_index, max_features)
    max_features = min(1000, embed_size) 
    model = BiLSTM(LR, dropout_val, embedding_matrix, max_features)
    history = LossHistory()
    
    training_history = model.fit(X_train, Y_train, batch_size=128, epochs=epochs,\
                             validation_data = (X_valid, Y_valid),\
                             class_weight = class_weight, callbacks=[EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001), history], verbose=1)
    
    return np.average(history.losses)
   
    
def hpo_monitor(study, trial):
    """
    Save optuna hpo study
    """
    joblib.dump(study, "temp_checkpoint.pkl")
    os.rename('temp_checkpoint.pkl', 'hpo_bilstm_checkpoint.pkl')
    
    
def get_best_params(best):
    """
    Saves best parameters of Optuna Study.
    """
    
    parameters = {}
    parameters["trial_id"] = best.number
    parameters["value"] = best.value
    parameters["params"] = best.params
    
    f = open("best_bilstm_hpo_params.txt","w")
    f.write(str(parameters))
    f.close()
    
def load_study():
    """
    Creates a new study or loads an existing study
    """
    
    try:
        STUDY = joblib.load("hpo_crisis_bilstm.pkl")
        print("Successfully loaded the existing study!")
        
        rem_trials = TRIALS - len(STUDY.trials_dataframe())
        
        if rem_trials > 0:
            STUDY.optimize(objective, n_trials=rem_trials, callbacks=[hpo_monitor])
        else:
            print("All trials done!")
        
    except Exception as e:
        print(e)
        print("Creating a new study!")
        
        STUDY = optuna.create_study(study_name='crisis-computing')
        STUDY.optimize(objective, n_trials=TRIALS, callbacks=[hpo_monitor])

    best_trial = STUDY.best_trial
    get_best_params(best_trial)
    
    return

def main():
    
    global TRIALS
    parser = argparse.ArgumentParser(description="Crisis Computing Workflow")
    parser.add_argument('--trials', type=int, default=1, help="Enter number of trials to perform HPO")
    args = parser.parse_args()
    
    TRIALS = args.trials
    load_study()
    print("Done with the program!!")
    return

if __name__ == "__main__":
    main()
