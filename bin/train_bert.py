#!/usr/bin/env python3

import torch, os
import numpy as np
import pandas as pd
import re, time
from sklearn.metrics import accuracy_score
import torch.nn as nn
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import DistilBertTokenizerFast
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['TRANSFORMERS_CACHE'] = '/tmp'

DATA_PATH = ""
BATCH_SIZE = 8
EPOCHS = 1
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CKPT = 'bert_final_model.pth'

def get_data():
    """
    returns the train, val and test tweets along with their classification labels.
    """

    train_df = pd.read_csv(DATA_PATH + 'preprocessed_train_tweets.csv')
    valid_df = pd.read_csv(DATA_PATH + 'preprocessed_val_tweets.csv')
    test_df  = pd.read_csv(DATA_PATH + 'preprocessed_test_tweets.csv')

    X_train = train_df['tweet_text']
    Y_train = train_df['text_info']

    X_valid = valid_df['tweet_text']
    Y_valid = valid_df['text_info']

    X_test = test_df['tweet_text']
    Y_test = test_df['text_info']

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def tokenize_and_map_ID(all_tweets):
    """
    Tokenize all of the sentences and map the tokens to thier word IDs.
    :params: all_tweets = combined tweet dataset
    """
    input_ids = []
    attention_masks = []

    for sent in all_tweets:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    return input_ids, attention_masks


def get_train_loader(input_ids, attention_masks, labels):
    """
    returns train dataset loader.
    """
    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    train_dataloader = DataLoader(
                train_dataset, 
                sampler = RandomSampler(train_dataset), 
                batch_size = BATCH_SIZE,
                drop_last = True
            )
    
    return train_dataloader


def get_test_loader(input_ids, attention_masks, labels):
    """
    returns train dataset loader.
    """

    test_dataset = TensorDataset(input_ids, attention_masks, labels)
    test_dataloader = DataLoader(
                test_dataset, 
                sampler = SequentialSampler(test_dataset), 
                batch_size = BATCH_SIZE 
            )
    
    return test_dataloader


def train_model(train_dataloader, test_dataloader):
    """
    returns trained BERT model, losses for train and test and accuracies
    :params: train_dataloader, test_dataloader = train and test dataset loaders
    """

    model = BertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels = 2,
        output_attentions = True,
        output_hidden_states = True, 
        cache_dir="/tmp"
    )

    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

    total_steps = len(train_dataloader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    pos_weight = torch.tensor([1.48, 0.75]).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    softmax_layer = nn.Softmax(dim=1)
    
    for epoch in range(EPOCHS):

        print("\nRunning Epoch: {}".format(epoch+1))
        start = time.time()
        total_train_loss = 0
        total_test_accuracy = 0
        total_test_loss = 0

        model.train()
        for batch in train_dataloader:

            # `batch` contains three pytorch tensors: [0]: input ids ,  [1]: attention masks, [2]: labels
            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)
  
            model.zero_grad()        

            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            b_labels = torch.nn.functional.one_hot(b_labels, num_classes=2).type_as(logits[0]).to(DEVICE)
            
            loss = criterion(logits[0], b_labels)

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)            

        model.eval()
        with torch.no_grad():

            for batch in test_dataloader:

                b_input_ids = batch[0].to(DEVICE)
                b_input_mask = batch[1].to(DEVICE)
                b_labels = batch[2].to(DEVICE)
                
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                
                b1_labels =torch.nn.functional.one_hot(b_labels, num_classes=2).type_as(logits[0]).to(DEVICE)
                loss = criterion(logits[0], b1_labels)

                total_test_loss += loss.item()
                output = softmax_layer(logits[0])
            
                label_ids = b_labels.to('cpu').numpy()
                y_pred = torch.argmax(output, dim=1).cpu().numpy()
                acc = accuracy_score(label_ids, y_pred)

                total_test_accuracy += acc

        avg_test_accuracy = total_test_accuracy / len(test_dataloader)
        avg_test_loss = total_test_loss / len(test_dataloader)
        

        print("\nTraining Loss: {0:.2f}".format(avg_train_loss))
        print("Testing Loss: {0:.2f}".format(avg_test_loss))
        print("Testing Accuracy: {0:.2f}".format(avg_test_accuracy))
        print("Time Taken: {0:.2f}".format((time.time()-start)/60))
        print("----------------------------------------------")
        
    print("\nTraining complete!")

    return model


if __name__ == "__main__":

    # initialize the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased',  do_lower_case=True, cache_dir="/tmp")
    print("model loaded")
    # get train, test and val data and labels
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = get_data()

    train_texts = list(X_train.values)
    val_texts = list(X_valid.values)
    test_texts = list(X_test.values)

    train_labels = Y_train
    val_labels   = Y_valid
    test_labels  = Y_test

 
    # record the size to split the dataset
    x = len(train_texts)
    y = len(train_texts) + len(val_texts)
    z = len(train_texts) + len(val_texts) + len(test_texts)

    # combine dataset
    all_tweets = train_texts + val_texts + test_texts
    all_labels = list(train_labels.values) + list(val_labels.values) + list(test_labels.values)


    # get tokenized word IDs and attention masks
    input_ids, attention_masks = tokenize_and_map_ID(all_tweets)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(all_labels)

    # Combine the training inputs into a TensorDataset.
    train_dataloader = get_train_loader(input_ids[:y], attention_masks[:y], labels[:y])
    test_dataloader = get_test_loader(input_ids[y:], attention_masks[y:], labels[y:])

    model = train_model(train_dataloader, test_dataloader)

    torch.save(model.state_dict(), MODEL_CKPT)  
    print("done!")
