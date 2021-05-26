#!/usr/bin/env python3

import torch, itertools
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


TRAIN_TEXT_PATH = 'bert_train_embeddings.csv'
TEST_TEXT_PATH = 'bert_test_embeddings.csv'

TRAIN_IMAGE_PATH = 'supcon_train_embeddings.csv'
TEST_IMAGE_PATH = 'supcon_test_embeddings.csv'

BATCH_SIZE = 32
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3
EPOCHS = 1
confusion_matrix_EF = 'early_fusion_MLP.png'
report_EF = 'early_fusion_MLP.csv'

def combine_representations(tweets, images):
    """
    combines 128-dim embeddings of images and 64-dim embeddings of tweets by matching tweet and image id.
    """

    data = pd.DataFrame(columns=["ID", "tweet_embedding", "image_embedding", "joint_representation", 'actual_class'])

    images['image_id'] = images.image_id.astype(str)
    tweets['tweet_id'] = tweets.tweet_id.astype(str)
    tweets['embedding'] = tweets.embedding.astype(str)

   
    for ind, image_row in images.iterrows():

        img_id = image_row['image_id']
        text_row = tweets.loc[tweets['tweet_id']==img_id]
        
       
        txt_emb = text_row['embedding'].values[0]
        img_emb = eval(image_row['embedding'])
        txt_emb = eval(txt_emb)

        joint = txt_emb + img_emb
            
        new_row = pd.Series([image_row['image_id'], text_row['embedding'].values[0], image_row['embedding'], joint, image_row['actual_class']], index=data.columns)
        data = data.append(new_row, ignore_index=True)
        
    return data



def plot_confusion_matrix(actual, predictions, title, file_name, report_name, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    normalize = True
    classes = ["Non-Inforamtive", "Informative"]
    cm =  confusion_matrix(actual, predictions)
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
    plt.savefig(file_name)
    
    report = pd.DataFrame(classification_report(actual, predictions, labels= [0, 1], target_names=["Non-Inforamtive", "Informative"], output_dict = True))
    report.to_csv(report_name)


class MLP(torch.nn.Module):
    """
    MLP Architecture
    """
    def __init__(self, input_size=192):
        super(MLP,self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self,x):
        output = self.model(x)
        return output


class Dataloader(torch.utils.data.Dataset):
    def __init__(self,data, label):
        self.feat = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        feat = self.feat.iloc[idx]
        label = self.label.iloc[idx]

        return torch.tensor(feat).type(torch.FloatTensor), torch.tensor(label).type(torch.FloatTensor)


def create_dataloaders(train_data, test_data):
    """
    returns train and test data-loaders to perform inference on.
    params: train_data, test_data: train and test data in the format [feat, labels] dataframe.
    """
    train_dloader = Dataloader(train_data['joint_representation'], train_data['actual_class'])
    test_dloader = Dataloader(test_data['joint_representation'], test_data['actual_class'])

    train_loader = torch.utils.data.DataLoader(train_dloader, shuffle=True, batch_size=BATCH_SIZE, num_workers=2, drop_last = True)
    test_loader = torch.utils.data.DataLoader(test_dloader, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)

    return train_loader, test_loader


def train_model(train_loader, test_loader):

    losses = {'train':[], 'test':[]}

    model = MLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9,0.999))
    criterion = torch.nn.BCELoss().to(DEVICE)

    for epoch in range(EPOCHS):

        print("Running Epoch: {}".format(epoch+1))
        epoch_train_loss = 0
        epoch_test_loss = 0
        model.train()

        for (feat, label) in train_loader:

            feat = feat.to(DEVICE)
            label = label.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(feat)
            
            loss = criterion(output.squeeze(), label)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            
        train_loss = epoch_train_loss/len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (feat, label) in test_loader:

                feat = feat.to(DEVICE)
                label = label.to(DEVICE)
                output = model(feat)

                v_loss = criterion(output.squeeze(), label)
                epoch_test_loss+=v_loss

                predictions = (output.squeeze()>0.5).cpu().float()
                correct += (predictions == label.cpu()).sum().item()
                total += len(label)

        test_loss = epoch_test_loss/len(test_loader)
        acc = correct/total
        print("Train loss: {0:.2f} Test loss: {1:0.2f}".format(train_loss, test_loss))
        print("Test Accuracy: {0:0.2f}".format(acc))
        print("---------------------------------------------------------")
        losses['train'].append(train_loss)
        losses['test'].append(test_loss)

    print("Training done!")
    return model, losses 

def test(model, test_loader):
    """
    performs forward pass on test set.
    """
    model.eval()
    predictions = []
    actual = []
    with torch.no_grad():
        for (feat, label) in test_loader:
            feat = feat.to(DEVICE)
            output = model(feat)
            actual.extend(label.cpu().tolist())
            pred = (output.squeeze()>0.5).cpu().float()
            correct = (pred == label).sum().item()
            predictions.extend(pred.tolist())

    return actual, predictions

if __name__ == "__main__":

    train_tweets = pd.read_csv(TRAIN_TEXT_PATH)
    test_tweets = pd.read_csv(TEST_TEXT_PATH)

    train_images = pd.read_csv(TRAIN_IMAGE_PATH)
    test_images = pd.read_csv(TEST_IMAGE_PATH)
   
    train_df = combine_representations(train_tweets, train_images)
    test_df = combine_representations(test_tweets, test_images)


    train_loader, test_loader = create_dataloaders(train_df[['joint_representation', 'actual_class']], test_df[['joint_representation', 'actual_class']])

    model, losses = train_model(train_loader, test_loader)
    
    actual, predicted = test(model, test_loader)
    
    plot_confusion_matrix(actual, predicted, "Early Fusion", confusion_matrix_EF, report_EF)