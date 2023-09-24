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

# Set random seeds for reproducibility
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)

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
            'PA': 0,
            'AP': 1,
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

train_batch_size = 256 # may need to reduce batch size if OOM error occurs
test_batch_size = 256


# train_path = '../test/train_sub_df.csv'
# val_path = '../test/validation_sub_df.csv'

train_path = '../train_df.csv'
val_path = '../validation_df.csv'

train_dataset = DatasetGenerator(train_path, '../../', 'AP/PA', transform=train_transform) 
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=32)

validate_dataset = DatasetGenerator(val_path,  '../../', 'AP/PA', transform=validate_transform)
validate_loader = DataLoader(validate_dataset, batch_size=test_batch_size, shuffle=False, num_workers=32)

# Create an instance of the ResNet-34 model
model = resnet34(pretrained=True)
num_classes = 1  # Replace with the number of classes in your task
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.cuda()

learning_rate = 1e-3
momentum_val=0.9
decay_val= 0.0

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay_val)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=1e-5, verbose=True)
criterion = nn.BCEWithLogitsLoss()  # Use appropriate loss function here

record_wb = True
if record_wb == True: 
        wandb.init(
                # set the wandb project where this run will be logged
                project="chexnet-" + 'pa' + "-pred",
                # track hyperparameters and run metadata
                config={"architecture": "ResNet34",
                        "dataset": "CheXpert",
                        "fine-tuned": True, 
                        "package": 'PyTorch',
                        "target": 'ap/pa', 
                        "data-subset": True
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
        labels = labels.cuda().float()

        optimizer.zero_grad()
        outputs = model(inputs).squeeze(dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        probs = torch.nn.functional.sigmoid(outputs)
        predicted_classes = (probs > 0.5).float()  # Convert to binary predictions (0 or 1)
        train_total += labels.size(0)
        train_correct += torch.sum(predicted_classes == labels).item()

    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validate_loader:
            inputs = inputs.cuda()
            labels = labels.cuda().float()
            
            outputs = model(inputs).squeeze(dim=1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            probs = torch.nn.functional.sigmoid(outputs)
            predicted_classes = (probs > 0.5).float()  # Convert to binary predictions (0 or 1)
            val_total += labels.size(0)
            val_correct += torch.sum(predicted_classes == labels).item()

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
        arc_name = 'CHEXPERT_AP_RESNET34_PY'
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
