#!/usr/bin/env python3

import numpy as np
import torch
import glob
import sys
from torch.utils.data import DataLoader, Dataset
import torchvision
import re
from PIL import Image
import os
import optuna
import joblib
import argparse

os.environ['MPLCONFIGDIR'] = '/tmp'
# Constant variables

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
MEAN = 0.4905, 0.4729, 0.4560 
STD = 0.2503, 0.2425, 0.2452
BATCH_SIZE = 2
EPOCHS = 2
PATIENCE = 6

# Paths
CHECKPOINT_DIR = ''
CHECKPOINT = 'resnet_final_model.pth'
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

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=4, verbose=False, delta=0, path= CHECKPOINT):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'early_stopping_vgg16model.pth'
        """
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """
        saves the current best version of the model if there is decrease in validation loss
        """
        torch.save(model.state_dict(), self.path)
        self.vall_loss_min = val_loss

# --------------------------------------------------------------------------------------------------------------------------

def train_loop(model, t_dataset, v_dataset, criterion, optimizer):
    """
    returns loss and accuracy of the model for 1 epoch.
    params: model -  resnet50
          dataset - train or val dataset
          flag - "train" for training, "val" for validation
          criterion - loss function
          optimizer - Adam optimizer
    """
    total = 0
    correct = 0
    epoch_t_loss = 0
    epoch_v_loss = 0
    model.train()
    
    for ind, (image, label) in enumerate(t_dataset):
        
        image = image.to(DEVICE)
        label = label.type(torch.float).to(DEVICE)

        optimizer.zero_grad()

        _, output = model(image)

        loss = criterion(output, label.unsqueeze(1))
        epoch_t_loss += loss.item()
        predicted = torch.round(output).squeeze(-1) 
        total += label.size(0)
        correct += (predicted==label).sum().item()

        loss.backward()
        optimizer.step()
      
    epoch_t_accuracy = 100*correct/total
    epoch_t_loss = epoch_t_loss/len(t_dataset)
    
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for ind, (image, label) in enumerate(v_dataset):
            image = image.to(DEVICE)
            label = label.type(torch.float).to(DEVICE)


            _, output = model(image)

            loss = criterion(output, label.unsqueeze(1))
            epoch_v_loss += loss.item()
            predicted = torch.round(output).squeeze(-1) 
            total += label.size(0)
            correct += (predicted==label).sum().item()


    epoch_v_accuracy = 100*correct/total
    epoch_v_loss = epoch_v_loss/len(v_dataset)
    
    return epoch_t_loss, epoch_t_accuracy, epoch_v_loss, epoch_v_accuracy


def train(train, val, trial):
    """
    returns train and validation losses of the model over complete training.
    params: train - train dataset
          val - validation dataset
          optimizer - optimizer for training
          criterion - loss function
    """
    
    loss = {}
    accuracy = {}
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    
    LR = trial.suggest_categorical("LR", [1e-3, 1e-4, 1e-5])
    
    model = Resnet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()
    early_stop = EarlyStopping(patience=PATIENCE)
    
    for epoch in range(EPOCHS):

        print("Running Epoch {}".format(epoch+1))

        epoch_train_loss, train_accuracy, epoch_val_loss, val_accuracy = train_loop( model, train, val, criterion, optimizer)
        train_losses.append(epoch_train_loss)
        train_acc.append(train_accuracy)
        val_losses.append(epoch_val_loss)
        val_acc.append(val_accuracy)

        print("Training loss: {0:.4f}  Train Accuracy: {1:0.2f}".format(epoch_train_loss, train_accuracy))
        print("Val loss: {0:.4f}  Val Accuracy: {1:0.2f}".format(epoch_val_loss, val_accuracy))
        print("--------------------------------------------------------")
        early_stop(epoch_val_loss, model)
        
        if early_stop.early_stop:
            print("Early stopping")
            break
            
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    print("Training done!")
    
    loss['train'] = train_losses
    loss['val'] = val_losses
    accuracy['train'] = train_acc
    accuracy['val'] = val_acc
    
    return loss, accuracy, model

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
     
        if self.transform:
            image = self.transform(image)

        return image, label


#--------------------------------------------------------------------------------------------------------------------------

def transforms_train():
    """
    Returns transformations on training dataset.
    params: mean - channel-wise mean of data
            std - channel-wise std of data
    """
    transfrms = []
    p = np.random.uniform(0, 1)

    transfrms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    
    if p >= 0.4 and p <=0.6:
        transfrms.append(torchvision.transforms.ColorJitter(0.2,0.1,0.2))

    transfrms.append(torchvision.transforms.ToTensor())  
    transfrms.append(torchvision.transforms.Normalize(MEAN, STD))
    
    return torchvision.transforms.Compose(transfrms)
    

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
        dataset = glob.glob(PATH + 'resized_train_*.png') + glob.glob(PATH + 'resized_train_*.jpg')
    
    elif data_type == 'val':
        dataset = glob.glob(PATH + 'resized_val_*.png') + glob.glob(PATH + 'resized_val_*.jpg')
        
    elif data_type == 'test':
        dataset = glob.glob(PATH + 'resized_test_*.png') + glob.glob(PATH + 'resized_test_*.jpg')
    
    print("Size of dataset ",data_type," is : ",len(dataset))
    
    data = DatasetLoader(dataset, transformation)

    dataset_loader = torch.utils.data.DataLoader(data, batch_size = BATCH_SIZE, shuffle=True, num_workers=4)

    return dataset_loader

#--------------------------------------------------------------------------------------------------------------------------


def objective(trial):
    
    print("Performing trial {}".format(trial.number))
    
    train_transform = transforms_train()
    test_transform  = transforms_test()
    
    train_data = get_dataloader('train', train_transform)
    val_data = get_dataloader('val', test_transform)
    test_data = get_dataloader('test', test_transform)

    loss, accuracy, model = train(train_data, val_data, trial)
    
    return np.average(loss['val'])
    
    
def hpo_monitor(study, trial):
    """
    Save optuna hpo study
    """
    joblib.dump(study, "temp.pkl")
    os.rename('temp.pkl', 'checkpoint_hpo.pkl')
    
    
def get_best_params(best):
    """
    Saves best parameters of Optuna Study.
    """
    
    parameters = {}
    parameters["trial_id"] = best.number
    parameters["value"] = best.value
    parameters["params"] = best.params
    
    f = open(CHECKPOINT_DIR+"best_resnet_hpo_params.txt","w")
    f.write(str(parameters))
    f.close()
    
    
def load_study():
    """
    Creates a new study or loads an existing study
    """
    
    try:
        STUDY = joblib.load(CHECKPOINT_DIR +"checkpoint_hpo.pkl")
        print("Successfully loaded the existing study!")
        
        rem_trials = TRIALS - len(STUDY.trials_dataframe())
        
        if rem_trials > 0:
            STUDY.optimize(objective, n_trials=rem_trials, callbacks=[hpo_monitor])
        else:
            print("All trials done!")
            pass
        
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
    parser.add_argument('--trials', type=int, default=2, help="Enter number of trials to perform HPO")
    args = parser.parse_args()
    
    TRIALS = args.trials
    load_study()
    
    return

if __name__ == "__main__":
    
    main()