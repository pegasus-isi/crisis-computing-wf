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
from resnet_big import SupConResNet

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
MEAN = 0.4905, 0.4729, 0.4560 
STD = 0.2503, 0.2425, 0.2452
OUT_FILE_TRAIN = 'train_supcon_embeddings.csv'
OUT_FILE_TEST = 'test_supcon_embeddings.csv'
CHECKPOINT = 'supcon_final_model.pth'
BATCH_SIZE = 1
CACHE = {}
PATH =  ""

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
        image_id = int(re.findall(r'[0-9]+', self.data[idx])[0])
                    
        if self.transform:
            image = self.transform(image)

        return image_id, image, label


def transforms_test():
    """
    Returns transformations on testing dataset.
    params: mean - channel-wise mean of data
            std - channel-wise std of data
    """

    test_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((600,600)),
                                                    torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(MEAN, STD)])
    return test_transform


def get_dataloader(data_type, transformation=None):
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

    dataset_loader = torch.utils.data.DataLoader(data, batch_size = BATCH_SIZE, shuffle=False, num_workers=2)
    

    return dataset_loader


def generate_embeddings(model, data_loader, dtype):
    """
    Generates 2048-dim embeddings from SupCon model.
    params: model - trained model 
            data_loader - train/test data
    """
    if dtype == 'train':
        out_file = open(OUT_FILE_TRAIN, 'w+')
    else:
        out_file = open(OUT_FILE_TEST, 'w+')
        
    csv_ob = csv.writer(out_file, delimiter=',')
    csv_ob.writerow(['image_id', 'embedding', 'actual_class'])
    
    model.eval()
    
    
    with torch.no_grad():  
        for image_id, image, label in data_loader:
           
            if str(image_id) in CACHE.keys():
                continue
            CACHE[str(image_id)] = 1
            
            image = image.to(DEVICE)
            label = label.type(torch.float).to(DEVICE)
            
            features = model(image)
            
            csv_ob.writerow([image_id.cpu().item(), features[0].cpu().tolist(), label.cpu().item()])

    return 

def main():

    transformation = transforms_test()
    train_loader = get_dataloader('train', transformation)
    test_loader = get_dataloader('test', transformation)
    model = SupConResNet('resnet50').to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT))
    generate_embeddings(model, train_loader, 'train')
    generate_embeddings(model, test_loader, 'test')
    return

if __name__ == '__main__':
    main()