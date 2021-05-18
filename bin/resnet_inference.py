#!/usr/bin/env python3

import numpy as np
import torch
import re
import glob
import sys
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import os
import csv
import re
import itertools
from sklearn.metrics import confusion_matrix

os.environ['MPLCONFIGDIR'] = '/tmp'
# Constant variables

DEVICE = "cpu" #("cuda" if torch.cuda.is_available() else "cpu")
MEAN = 0.4905, 0.4729, 0.4560 
STD = 0.2503, 0.2425, 0.2452
EPOCHS = 1

# Paths

CHECKPOINT = 'resnet_final_model.pth'
OUT_FILE_TRAIN = 'resnet_train_output.csv'
OUT_FILE_TEST = 'resnet_test_output.csv'


PATH = ""
#--------------------------------------------------------------------------------------------------------------------------

class Resnet(torch.nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        
        modules = list(self.resnet.children())[:-1]
        self.resnet = torch.nn.Sequential(*modules)
 
        for params in self.resnet.parameters():
            params.requires_grad = False
    

        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=2048, out_features=1024, bias=True),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(1024, 1, bias=True),
            torch.nn.Sigmoid())


    def forward(self, x):
        feat = self.resnet(x)
        output = self.head(feat)
        return feat, output
    
#--------------------------------------------------------------------------------------------------------------------------

def run_inference(model, data, dtype):
    """
    returns output probabilites and prediction classes
    params: model - model for testing
          test - test dataset
    """
    if dtype == 'train':
        out_file = open(OUT_FILE_TRAIN, 'w+')
    else:
        out_file = open(OUT_FILE_TEST, 'w+')
        
    csv_ob = csv.writer(out_file, delimiter=',')
    csv_ob.writerow(['image_id', 'predicted_score', 'predicted_class', 'actual_class'])
    
    model.eval()
    actual = []
    preds = []
    
    with torch.no_grad():  
        for image_id, image, label in data:
            print(image_id)
            image = image.to(DEVICE)
            label = label.type(torch.float).to(DEVICE)
            _, output_prob = model(image)
            predicted = torch.round(output_prob).squeeze(-1)
            actual.append(label.cpu().item())
            preds.append(predicted.cpu().item())
               
            csv_ob.writerow([image_id[0] ,output_prob.cpu().item(), int(predicted.cpu().item()), int(label.cpu().item())])
    
    if dtype == 'test':
        plot_confusion_matrix(actual, preds)
    return

def plot_confusion_matrix(actual, predictions, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    normalize = True
    classes = ["Informative", "Noninforamtive"]
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
    plt.savefig("resnet_confusion_matrix.png")
    
#--------------------------------------------------------------------------------------------------------------------------

class DatasetLoader(Dataset):
    def __init__(self, data, transform = None):
        
        self.data = data
        self.data_len = len(data)
        self.transform = transform

    def __len__(self):
        
        return self.data_len

    def __getitem__(self, idx):
       
        image = Image.open(self.data[idx]).convert('RGB')
 
        label = int(re.findall(r'[0-9]+', self.data[idx])[-1])
        image_id = (re.findall(r'[0-9]+', self.data[idx]))
        img_id = image_id[0]+ '_' + image_id[1]

        if self.transform:
            image = self.transform(image)

        return img_id, image, label


#--------------------------------------------------------------------------------------------------------------------------

def transforms_test():
    """
    Returns transformations on testing dataset.
    params: mean - channel-wise mean of data
            std - channel-wise std of data
    """

    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(MEAN, STD)])
    return test_transform


def get_dataloader(data_type, transformation):
    """
    returns train, validation and test dataloafer objects
    params: train_transform - Augmentation for trainset
            test_transform - Augmentation for testset
            batch_size - size of batch
            n_workers - number of workers
    """
    if data_type == 'train':
        dataset = glob.glob(PATH + '*train_*.png') + glob.glob(PATH + '*train_*.jpg') + glob.glob(PATH + '*val_*.png') + glob.glob(PATH + '*val_*.jpg')
        
    elif data_type == 'test':
        dataset = glob.glob(PATH + '*test_*.png') + glob.glob(PATH + '*test_*.jpg')
    
    data = DatasetLoader(dataset, transformation)

    dataset_loader = torch.utils.data.DataLoader(data, batch_size = 1, shuffle=False, num_workers=4)

    return dataset_loader

#--------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    test_transform  = transforms_test()
    train_data = get_dataloader('train', test_transform)
    test_data = get_dataloader('test', test_transform)

    model = Resnet().to(DEVICE)
    model.state_dict(torch.load(CHECKPOINT))
  
    run_inference(model, test_data, 'test')

    run_inference(model, train_data, 'train')
