#!/usr/bin/env python3

import torch, os
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import DistilBertTokenizerFast
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig


os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
DATA_PATH = ""
BATCH_SIZE = 64
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CHECKPOINT = 'bert_final_model.pth'
TRAIN_EMBEDDINGS = 'bert_train_embeddings.csv'
TEST_EMBEDDINGS = 'bert_test_embeddings.csv'


def get_data():
    """
    returns the train, val and test tweets along with their classification labels.
    """

    train_df = pd.read_csv(DATA_PATH + 'preprocessed_train_tweets.csv')
    valid_df = pd.read_csv(DATA_PATH + 'preprocessed_val_tweets.csv')
    test_df  = pd.read_csv(DATA_PATH + 'preprocessed_test_tweets.csv')

    X_train = train_df['tweet_text']
    Y_train = train_df['text_info']

    id_train = train_df['tweet_id']
    id_val = valid_df['tweet_id']
    id_test = test_df['tweet_id']
    
    X_valid = valid_df['tweet_text']
    Y_valid = valid_df['text_info']

    X_test = test_df['tweet_text']
    Y_test = test_df['text_info']

    train_ids = list(id_train.values) + list(id_val.values)
    test_ids = list(id_test.values)

    return train_ids, test_ids, X_train, X_valid, X_test, Y_train, Y_valid, Y_test


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
                sampler = SequentialSampler(train_dataset), 
                batch_size = BATCH_SIZE,
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

def get_embeddings(dataloader, model):
    """
    returns tweet embeddings.
    :params: dataloader = train/test dataset
             model = trained model
    """
    embed = []
    true_label = []
    model.eval()
    with torch.no_grad(): 

        for batch in dataloader:
            
            # `batch` contains three pytorch tensors: [0]: input ids ,  [1]: attention masks, [2]: labels
            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)
            true_label.extend(b_labels.cpu().tolist())
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            hidden_states = logits[2]
            hidden_states = hidden_states[0]
            hidden_states = hidden_states.permute(1,0,2,3)

            token_vecs = hidden_states[-2]

            sentence_embedding = torch.mean(token_vecs, dim=1)

            embed.extend(sentence_embedding.cpu().tolist())

    return embed, true_label

def generate_csv(ids, embeds, labels, fname):
    """
    generates csv containing tweet id, tweet embeddings and tweet labels.
    """
    data_csv = pd.DataFrame()
    data_csv['tweet_id'] = ids
    data_csv['embedding'] = embeds
    data_csv['actual_class'] = labels

    data_csv.to_csv(fname, index=False)

    return

if __name__ == "__main__":

    # initialize the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased',do_lower_case=True, cache_dir="/tmp")

    # get train, test and val data and labels
    train_ids, test_ids, X_train, X_valid, X_test, Y_train, Y_valid, Y_test = get_data()

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

    train_dataloader = get_train_loader(input_ids[:y], attention_masks[:y], labels[:y])
    test_dataloader = get_test_loader(input_ids[y:], attention_masks[y:], labels[y:])

    model = BertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels = 2,
        output_attentions = True,
        output_hidden_states = True, 
        cache_dir="/tmp"
    )
    
    model.load_state_dict(torch.load(MODEL_CHECKPOINT))
    model.to(DEVICE)
    train_embedding, train_true_class = get_embeddings(train_dataloader, model)
    test_embedding, test_true_class = get_embeddings(test_dataloader, model)


    generate_csv(train_ids, train_embedding, train_true_class, TRAIN_EMBEDDINGS)
    generate_csv(test_ids, test_embedding, test_true_class, TEST_EMBEDDINGS)
    print("done!")