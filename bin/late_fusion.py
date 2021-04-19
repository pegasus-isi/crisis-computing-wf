#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import torch
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import itertools
import warnings
warnings.filterwarnings('ignore')

PATH_TEXT_TRAIN  = 'bilstm_train_output.csv'
PATH_TEXT_TEST   = 'bilstm_test_output.csv'
PATH_IMAGE_TRAIN = 'resnet_train_output.csv'
PATH_IMAGE_TEST  = 'resnet_test_output.csv'

confusion_matrix_MPC = 'late_fusion_MPC.png'
confusion_matrix_LR = 'late_fusion_LR.png'
confusion_matrix_MLP = 'late_fusion_MLP.png'
report_MLP = 'late_fusion_MLP.csv'
report_MPC = 'late_fusion_MPC.csv'
report_LR = 'late_fusion_LR.csv'

BATCH_SIZE = 2

# ----------------------------------------------Mean Probability Concatenation-------------------------------------------------
def mean_prob_concatenation(test_data):
    """
    Class prediction probabilities obtained by both textual and visual
    modalities have been combined, averaged and then thresholded
    params: test_data: testing dataset
    returns fused output of image classification and text classification
    """
    fusion_op = (test_data['image_prob'] + test_data['text_prob']).div(2)
    fusion_op[fusion_op>0.5] = 1
    fusion_op[fusion_op<=0.5] = 0

    return fusion_op


# --------------------------------------------Logistic Regression Decision Policy----------------------------------------------

def Logistic_Regression_policy(train_data, test_data):
    """
    Logistic Regression Decision policy for late fusion
    """

    clf = LogisticRegression(random_state=0, solver='newton-cg').fit(train_data[["image_prob", "text_prob"]], list(train_data["actual_class"]) )
    predictions = clf.predict(test_data[["image_prob","text_prob"]])
    correct = (predictions == test_data['actual_class']).sum()
  
    return list(test_data['actual_class']), list(predictions)

# ------------------------------------------Multi-layered Perceptron Decision Policy-------------------------------------------


class MLP(torch.nn.Module):
    """
    MLP Architecture
    """
    def __init__(self, input_size=2, hidden_dim=128):
        super(MLP,self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self,x):
        output = self.model(x)
        return output

    
class Dataloader(torch.utils.data.Dataset):
    def __init__(self,data):
        self.feat = data[["image_prob", "text_prob"]]
        self.label = data["actual_class"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        feat = self.feat.iloc[idx].values
        label = self.label.iloc[idx]
        return feat, label


def create_dataloaders(train_data, test_data):
    """
    returns train and test data-loaders to perform inference on.
    params: train_data, test_data: train and test data in the format [feat, labels] dataframe.
    """
    train_loader = Dataloader(train_data.astype('float32'))
    test_loader = Dataloader(test_data.astype('float32'))

    train_loader = torch.utils.data.DataLoader(train_loader, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_loader, shuffle=True, batch_size=len(test_loader), num_workers=4)

    return train_loader, test_loader

def train_model(epochs, criterion, optimizer, model, train_loader):
    """
    Train the MLP architecture and return the model
    """
    training_loss = []
    for epoch in range(epochs):
        epoch_loss = 0
        for (feat, label) in train_loader:
            feat = feat.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = mlp(feat)
            loss = criterion(output.squeeze(), label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss/=len(train_loader)
        training_loss.append(epoch_loss)
        print("Epoch: {}".format(epoch+1))
        print("Train loss: {0:.2f}".format(epoch_loss))
        print("------------------------")
        
    print("Training done!")
    return model

def test(model, test_loader):
    """
    performs forward pass on test set.
    """
    model.eval()
    predictions = []
    actual = []
    with torch.no_grad():
        for (feat, label) in test_loader:
            feat = feat.cuda()
            output = model(feat)
            actual = label.cpu()
            predictions = (output.squeeze()>0.5).cpu().float()
            correct = (predictions == label).sum().item()
   
    return actual, predictions

# ---------------------------------------------Preprocessing and Statistics----------------------------------------------------


def generate_data(img_data, text_data):
    """
    returns train_data in format ['image_probabilities', 'text_probabilities', 'actual output label']
    params: text_data: text based dataset
          img_data: image based dataset
    """
    print("Length of image data: ",len(img_data))
    print("Length of text data: ",len(text_data))
    
    data = pd.DataFrame(columns=["image_prob", "text_prob", 'actual_class'])
    img_data['image_id'] = img_data.image_id.astype(str)
    text_data['tweet_id'] = text_data.tweet_id.astype(str)
   
    for ind, image_row in img_data.iterrows():
        text_row = text_data.loc[text_data['tweet_id']==image_row['image_id']]
        prob = pd.Series([image_row['predicted_score'], text_row['predicted_score'].values, image_row['actual_class']], index=data.columns)
        data = data.append(prob, ignore_index=True)

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

    
# -----------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    image_train = pd.read_csv(PATH_IMAGE_TRAIN)
    image_test = pd.read_csv(PATH_IMAGE_TEST)
    text_train = pd.read_csv(PATH_TEXT_TRAIN)
    text_test = pd.read_csv(PATH_TEXT_TEST)

    train_data = generate_data(image_train, text_train)       
    test_data = generate_data(image_test, text_test)
    
    # call to Mean probability Concatenation decision policy
    fusion_output = mean_prob_concatenation(test_data)
    
    # call to Logistic Regression decision policy
    LR_actual, LR_predicted = Logistic_Regression_policy(train_data, test_data)
    
    # call to MLP Decision Policy
    learning_rate = 0.01
    epochs = 10
    mlp = MLP().cuda()
    mlp.train()
    optimizer = torch.optim.Adam(mlp.parameters(), lr = learning_rate, betas=(0.9,0.999))
    criterion = torch.nn.BCELoss()
    train_loader, test_loader = create_dataloaders(train_data, test_data)
    trained_model = train_model(epochs, criterion, optimizer, mlp, train_loader)
    MLP_actual, MLP_predicted = test(trained_model, test_loader)


    # call to confusion matrix and report plot function
    plot_confusion_matrix(list(test_data['actual_class']), list(fusion_output), 'Mean Probability Concatenation', confusion_matrix_MPC, report_MPC)
    plot_confusion_matrix(LR_actual, LR_predicted, "Logistic Regression Decision Policy", confusion_matrix_LR, report_LR)
    plot_confusion_matrix(MLP_actual, MLP_predicted, "MLP Decision Policy", confusion_matrix_MLP, report_MLP)