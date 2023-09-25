import os
import numpy as np
import time 
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
from torch.utils.data import random_split, DataLoader

from sklearn.metrics import roc_auc_score
import pandas as pd
import wandb
import re

from models import DenseNet121
from models import DenseNet169
from models import DenseNet201
from dataset import DatasetGenerator

#-------------------------------------------------------------------------------- 

class ChexnetTrainer():

    #---- Train the densenet network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def train (pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):
        

        # Training loop
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="chexnet",
            
            # track hyperparameters and run metadata
            config={
            "architecture": "densenet121",
            "dataset": "ChexNet Subset",
            }
        )
    
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda()
                
        #-------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)
              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=5, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=5, pin_memory=True)
        
        # Define the size of the subset
        subset_size = 100

        # Calculate the size of the remaining data for training and validation
        remaining_train_size = len(datasetTrain) - subset_size
        remaining_val_size = len(datasetVal) - subset_size

        # Create the subset DataLoader for training data
        subset_dataset_train, _ = random_split(datasetTrain, [subset_size, remaining_train_size])
        subset_data_loader_train = DataLoader(dataset=subset_dataset_train, batch_size=trBatchSize, shuffle=True, num_workers=5, pin_memory=True)

        # Create the subset DataLoader for validation data
        subset_dataset_val, _ = random_split(datasetVal, [subset_size, remaining_val_size])
        subset_data_loader_val = DataLoader(dataset=subset_dataset_val, batch_size=trBatchSize, shuffle=False, num_workers=5, pin_memory=True)

        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        loss = torch.nn.BCELoss(size_average = True)
        
        #---- Load checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        #---- TRAIN THE NETWORK
        
        lossMIN = 100000
        
        for epochID in range (0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
                         
            ChexnetTrainer.epochTrain(model, dataLoaderTrain, optimizer, scheduler, epochID, nnClassCount, loss)
            lossVal, losstensor, val_Acc = ChexnetTrainer.epochVal(model, dataLoaderVal, optimizer, scheduler, epochID, nnClassCount, loss)

            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(losstensor.item())
            #wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": lossVal, "val_acc": val_Acc})

            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
            
        wandb.finish()

    #-------------------------------------------------------------------------------- 
       
    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()
        
        accValues = []
        correct_classes  = [0]*classCount
        class_sizes = [0]*classCount

        total_loss = 0
        lossNorm = 0

        for batchID, (input, target) in enumerate (dataLoader):
                        
            target = target.cuda()
                 
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(varInput)
            
            lossValue = loss(varOutput, varTarget)

            total_loss += lossValue
            lossNorm += 1
            
            correct, correct_class, class_size = ChexnetTrainer.computeAccuracy(varTarget, varOutput, classCount)
            accValues.append(correct)
            correct_classes += correct_class
            class_sizes += class_size

            optimizer.zero_grad()
            lossValue.backward()
            optimizer.step()

        outLoss = total_loss / lossNorm
        outAcc = np.mean(np.concatenate(accValues))
        classAcc = correct_classes / class_sizes

        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        #for i in range (0, len(classAcc)):
            #wandb.log({f"train acc ({CLASS_NAMES[i]})": classAcc[i]})

        wandb.log({"custom_step": epochMax, "train_loss": outLoss, "train_acc": outAcc})
        return outLoss, outAcc
            
    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval()
        
        accValues = []
        correct_classes  = [0]*classCount
        class_sizes = [0]*classCount

        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0

        for i, (input, target) in enumerate (dataLoader):
            
            target = target.cuda()
            
            with torch.no_grad():
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)
                varOutput = model(varInput)

            losstensor = loss(varOutput, varTarget)
            losstensorMean += losstensor
            
            lossVal += losstensor.item()
            lossValNorm += 1

            correct, correct_class, class_size = ChexnetTrainer.computeAccuracy(varTarget, varOutput, classCount)
            accValues.append(correct)
            correct_classes += correct_class
            class_sizes += class_size

        outLoss = lossVal / lossValNorm
        outAcc = np.mean(np.concatenate(accValues))
        losstensorMean = losstensorMean / lossValNorm

        classAcc = correct_classes / class_sizes
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        for i in range (0, classCount):
            # Create a dictionary for logging
            wandb.log({"custom_step": epochMax, f"val_acc ({CLASS_NAMES[i]})": classAcc[i]})
        wandb.log({"custom_step": epochMax, "val_loss": outLoss, "val_acc": outAcc})
        return outLoss, losstensorMean, outAcc
               
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes

    def computeAccuracy(dataGT, dataPred, classCount):
        
        # Convert predicted probabilities to predicted classes
        predicted_probs = dataPred.detach().cpu().numpy()
        true_labels = dataGT.detach().cpu().numpy()

        predicted_classes = np.argmax(predicted_probs, axis=1)

        # Convert true one-hot encoded labels to true classes
        true_classes = np.argmax(true_labels, axis=1)

        # Calculate accuracy for the batch
        matches = predicted_classes == true_classes

        # Initialize an array to store the number of matches per class
        class_match_counts = np.zeros(classCount, dtype=int)
        class_counts = np.zeros(classCount, dtype=int)
        
        for class_idx in range(classCount):
            class_matches = matches[true_classes == class_idx]
            class_match_counts[class_idx] = np.sum(class_matches)
            class_counts[class_idx] = len(matches[true_classes == class_idx])

        return matches, class_match_counts, class_counts
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC
        
        
    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        cudnn.benchmark = True

        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, None).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda()
        # Load the checkpoint
        checkpoint = torch.load(pathModel)
        
        print(f'After Checkpoint: {pathModel} -------------------------')
        state_dict = checkpoint['state_dict']
        remove_data_parallel = False # Change if you don't want to use nn.DataParallel(model)

        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            match = pattern.match(key)
            new_key = match.group(1) + match.group(2) if match else key
            new_key = new_key[7:] if remove_data_parallel else new_key
            state_dict[new_key] = state_dict[key]
            # Delete old key only if modified.
            if match or remove_data_parallel: 
                del state_dict[key]

        model.load_state_dict(state_dict)
        model.cuda()

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=5, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        
        for i, (input, target) in enumerate(dataLoaderTest):
            
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)
            
            bs, n_crops, c, h, w = input.size()
            
            with torch.no_grad(): 
                varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
            
            out = model(varInput)
            outMean = out.view(bs, n_crops, -1).mean(1)
            
            outPRED = torch.cat((outPRED, outMean.data), 0)

        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        

        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        
        # Create a DataFrame
        data = {'Class': CLASS_NAMES, 'AUROC Score': aurocIndividual}
        df = pd.DataFrame(data)

        # Add AUROC Mean as the first row
        df.loc[-1] = ['AUROC Mean', aurocMean]
        df.index = df.index + 1
        df = df.sort_index()
        df['Model'] = pathModel

        # Specify the CSV file path
        csv_file_path = 'auroc_scores.csv'

        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)

        print("AUROC scores saved to:", csv_file_path)

        return
#-------------------------------------------------------------------------------- 




