from datetime import datetime
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import wandb
import os
import pandas as pd
from PIL import Image
import random as python_random
import seaborn as sns
import sys
import math

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import auc, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet34
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import random_split, DataLoader
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.cuda.amp as amp 

# CheXpert images can be found: https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
data_df = pd.read_csv('train_cheXbert.csv')
demo_df = pd.DataFrame(pd.read_excel("CHEXPERT_DEMO.xlsx", engine='openpyxl')) #pip install openpyxl
split_df = pd.read_csv('chexpert_split_2021_08_20.csv').set_index('index')

data_df = pd.concat([data_df,split_df], axis=1)
data_df = data_df[~data_df.split.isna()]

path_split =  data_df.Path.str.split("/", expand = True)
data_df["patient_id"] = path_split[2]
demo_df = demo_df.rename(columns={'PATIENT': 'patient_id'})
data_df = data_df.merge(demo_df, on="patient_id")

mask = (data_df.PRIMARY_RACE.str.contains("Black", na=False))
data_df.loc[mask, "race"] = "BLACK/AFRICAN AMERICAN"

mask = (data_df.PRIMARY_RACE.str.contains("White", na=False))
data_df.loc[mask, "race"] = "WHITE"

mask = (data_df.PRIMARY_RACE.str.contains("Asian", na=False))
data_df.loc[mask, "race"] = "ASIAN"
data_df.split.value_counts(normalize=True)
data_df.race.value_counts(normalize=True)
data_df[['split', 'race']].value_counts(normalize=True)
train_df = data_df[data_df.split=="train"]
validation_df = data_df[data_df.split=="validate"]
test_df = data_df[data_df.split=="test"]

train_df.to_csv('train_df.csv')
validation_df.to_csv('validation_df.csv')
test_df.to_csv('test_df.csv')

# Set random seeds for reproducibility
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)

class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()
        self.resnet34 = resnet34(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet34.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc =  nn.Sequential(nn.Linear(512, num_classes), nn.Softmax(dim=1)  # Apply softmax for probability distribution
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define dataset and dataloaders
class DatasetGenerator(Dataset):
    def __init__(self, data_frame, root_dir, nnTarget, transform=None):
        self.data_frame = pd.read_csv(data_frame)
        self.target = nnTarget
        self.listImagePaths = list(root_dir + self.data_frame['Path'])
        self.listImageLabels = list(self.data_frame[nnTarget])
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')

        label = self.listImageLabels[index]
        # Define a dictionary to map class labels to class indices
        class_to_idx = {
            'ASIAN': 0,
            'WHITE': 1,
            'BLACK/AFRICAN AMERICAN': 2
        }
        imageLabel = class_to_idx[label]
        if self.transform != None: imageData = self.transform(imageData)

        return imageData, imageLabel

HEIGHT, WIDTH = 320, 320

train_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomRotation(15),      # Randomly rotate the image by up to 15 degrees
    transforms.RandomResizedCrop(320, scale=(0.9, 1.1)), 
    transforms.RandomHorizontalFlip(),  # Horizontal Flip
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Define preprocessing transformations
validate_transform = transforms.Compose([
    transforms.Resize((320, 320)),                   # Resize the input image to 256x256
    transforms.CenterCrop(320),               # Crop the center 224x224 portion of the image
    transforms.ToTensor(),                    # Convert to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

train_dataset = DatasetGenerator('train_df.csv', '../', 'race', transform=train_transform) 
validate_dataset = DatasetGenerator('validation_df.csv',  '../', 'race', transform=validate_transform)

learning_rate = 1e-3
momentum_val=0.9
decay_val= 0.0
train_batch_size = 256 # may need to reduce batch size if OOM error occurs
test_batch_size = 256

train_epoch = math.ceil(len(train_dataset) / train_batch_size)
val_epoch = math.ceil(len(validate_dataset) / test_batch_size)

# Create an instance of the ResNet-34 model
model = resnet34(pretrained=True)

# Modify the final fully connected layer for your specific task
num_classes = 3  # Replace with the number of classes in your task
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.cuda()

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=32)
validate_loader = DataLoader(validate_dataset, batch_size=test_batch_size, shuffle=False, num_workers=32)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay_val)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-5, verbose=True)
criterion = nn.CrossEntropyLoss()  # Use appropriate loss function here

record_wb = True
if record_wb == True: 
        wandb.init(
                # set the wandb project where this run will be logged
                project="chexnet-" + 'race' + "-pred",
                # track hyperparameters and run metadata
                config={"architecture": "ResNet34",
                        "dataset": "CheXpert",
                        "fine-tuned": True, 
                        "package": 'PyTorch',
                        "target": 'race', 
                        "data-subset": False
                }
                )

# Training loop
best_val_loss = float('inf')
best_val_accuracy = 0
best_model_weights = None

num_epochs = 100
no_improvement_count = 0  # Initialize a counter for early stopping
early_stopping_patience = 10  # Define the patience for early stopping

for epoch in range(num_epochs):
    train_loss = 0
    train_correct = 0
    train_total = 0 

    val_predictions = []  # Store predicted probabilities for AUROC and PR-AUC
    val_labels = []  # S

    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_classes = torch.max(probs, 1)
        train_total += labels.size(0)
        train_correct += torch.sum(predicted_classes==labels).item()

    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validate_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_classes = torch.max(probs, 1)

            val_total += labels.size(0)
            val_correct += torch.sum(predicted_classes==labels).item()

            # Collect predicted probabilities and labels for AUROC and PR-AUC
            val_predictions.extend(probs.cpu().numpy())  # Assuming you have 2 classes, using probabilities of class 1
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(validate_loader)
    val_accuracy = val_correct / val_total

    # Calculate AUROC and PR-AUC
    label_binarizer = LabelBinarizer().fit(val_labels)
    y_onehot_test = label_binarizer.transform(val_labels)

    auroc = roc_auc_score(y_onehot_test, val_predictions, multi_class='ovr')
    pr_auc = average_precision_score(y_onehot_test, val_predictions)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, \
          Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUROC: {auroc:.4f}, Val AUC: {pr_auc:.4f}')
    wandb.log({'epoch': epoch, 'Train Loss': train_loss, 'Train Accuracy': train_accuracy, 'Val Loss': val_loss, 'Val Accuracy': val_accuracy, 'Val AUROC': auroc, 'Val AUC': pr_auc})
    # Save the model if validation loss improves
    if (val_loss < best_val_loss) or (val_accuracy > best_val_accuracy):
        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        best_model_weights = model.state_dict()
        arc_name = 'CHEXPERT_RACE_RESNET34_PY'
        var_date = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = "models/" + str(arc_name) + "_" + var_date + f"_epoch:{epoch:03d}_val_loss:{val_loss:.2f}.pth.tar"
        torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': val_loss, 'optimizer' : optimizer.state_dict()}, model_name)
        print('model saved')
        no_improvement_count = 0  # Reset the counter when validation loss improves
    else:
        no_improvement_count += 1  # Increment the counter if there's no improvement

    # Check if early stopping criteria are met
    if no_improvement_count >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch + 1} as validation loss did not improve for {early_stopping_patience} consecutive epochs.')
        break  # Exit the loop if early stopping criteria are met

    scheduler.step(val_loss)

print("Training finished.")
wandb.finish()

# multilabel_predict_test = model.predict(test_batches, max_queue_size=10, verbose=1, steps=math.ceil(len(test_df)/test_batch_size), workers=16)
# result = multilabel_predict_test
# #result = model.predict(validate_batches, val_epoch)
# labels = np.argmax(result, axis=1)
# target_names = ['Asian', 'Black', 'White']

# print ('Classwise ROC AUC \n')
# for p in list(set(labels)):
#     fpr, tpr, thresholds = roc_curve(test_batches.classes, result[:,p], pos_label = p)
#     auroc = round(auc(fpr, tpr), 2)
#     print ('Class - {} ROC-AUC- {}'.format(target_names[p], auroc))

# print (classification_report(test_batches.classes, labels, target_names=target_names))
# class_matrix = confusion_matrix(test_batches.classes, labels)

# sns.heatmap(class_matrix, annot=True, fmt='d', cmap='Blues')