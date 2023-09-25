import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl
import wandb
import datetime

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torchmetrics.classification import Accuracy, MulticlassAccuracy, AUROC
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser

image_size = (224, 224)
num_classes_disease = 14
num_classes_sex = 2
num_classes_race = 3
class_weights_race = (1.0, 1.0, 1.0) # can be changed to balance accuracy
batch_size = 150
epochs = 20
num_workers = 4
img_data_dir = '../../data/'

class WandbCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        # Log training and validation loss to Wandb
        wandb.log({"train_loss": trainer.callback_metrics["train_loss"],
                   "val_loss": trainer.callback_metrics["val_loss"]})

class CheXpertDataset(Dataset):
    def __init__(self, csv_file_img, image_size, resize=True, augmentation=False, pseudo_rgb = True):
        self.data = pd.read_csv(csv_file_img)
        self.image_size = image_size
        self.do_augment = augmentation
        self.do_resize = resize
        self.pseudo_rgb = pseudo_rgb

        self.labels = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

        self.resize = T.Compose([
            T.Resize((224,224), antialias=True)
        ])

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = img_data_dir + self.data.loc[idx, 'Path']
            img_label_disease = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label_disease[i] = np.array(self.data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            img_label_sex = np.array(self.data.loc[idx, 'sex_label'], dtype='int64')
            img_label_race = np.array(self.data.loc[idx, 'race_label'], dtype='int64')

            sample = {'image_path': img_path, 'label_disease': img_label_disease, 'label_sex': img_label_sex, 'label_race': img_label_race}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image']).unsqueeze(0)
        label_disease = torch.from_numpy(sample['label_disease'])
        label_sex = torch.from_numpy(sample['label_sex'])
        label_race = torch.from_numpy(sample['label_race'])

        if self.do_resize:
            image = self.resize(image)

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {'image': image, 'label_disease': label_disease, 'label_sex': label_sex, 'label_race': label_race}

    def get_sample(self, item):
        sample = self.samples[item]
        image = imread(sample['image_path']).astype(np.float32)

        return {'image': image, 'label_disease': sample['label_disease'], 'label_sex': sample['label_sex'], 'label_race': sample['label_race']}


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, csv_train_img, csv_val_img, csv_test_img, image_size, pseudo_rgb, batch_size, num_workers):
        super().__init__()
        self.csv_train_img = csv_train_img
        self.csv_val_img = csv_val_img
        self.csv_test_img = csv_test_img
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = CheXpertDataset(self.csv_train_img, self.image_size, augmentation=True, pseudo_rgb=pseudo_rgb)
        self.val_set = CheXpertDataset(self.csv_val_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
        self.test_set = CheXpertDataset(self.csv_test_img, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

class ResNet(pl.LightningModule):
    def __init__(self, num_classes_disease, num_classes_sex, num_classes_race, class_weights_race):
        super().__init__()
        self.num_classes_disease = num_classes_disease
        self.num_classes_sex = num_classes_sex
        self.num_classes_race = num_classes_race
        self.class_weights_race = torch.FloatTensor(class_weights_race)
        self.backbone = models.resnet34(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes_disease)
        self.fc_sex = nn.Linear(num_features, self.num_classes_sex)
        self.fc_race = nn.Linear(num_features, self.num_classes_race)
        self.fc_connect = nn.Identity(num_features)
        self.backbone.fc = self.fc_connect

    def forward(self, x):
        embedding = self.backbone.forward(x)
        out_disease = self.fc_disease(embedding)
        out_sex = self.fc_sex(embedding)
        out_race = self.fc_race(embedding)
        return out_disease, out_sex, out_race

    def configure_optimizers(self):
        params_backbone = list(self.backbone.parameters())
        params_disease = params_backbone + list(self.fc_disease.parameters())
        params_sex = params_backbone + list(self.fc_sex.parameters())
        params_race = params_backbone + list(self.fc_race.parameters())
        optim_disease = torch.optim.Adam(params_disease, lr=0.001)
        optim_sex = torch.optim.Adam(params_sex, lr=0.001)
        optim_race = torch.optim.Adam(params_race, lr=0.001)
        return optim_disease, optim_sex, optim_race

    def unpack_batch(self, batch):
        return batch['image'], batch['label_disease'], batch['label_sex'], batch['label_race']

    def process_batch(self, batch):
        img, lab_disease, lab_sex, lab_race = self.unpack_batch(batch)
        out_disease, out_sex, out_race = self.forward(img)
        loss_disease = F.binary_cross_entropy(torch.sigmoid(out_disease), lab_disease)
        loss_sex = F.cross_entropy(out_sex, lab_sex)
        loss_race = F.cross_entropy(out_race, lab_race, weight=self.class_weights_race.type_as(img))
        return loss_disease, loss_sex, loss_race

    def training_step(self, batch, batch_idx): #, optimizer_idx):
        loss_disease, loss_sex, loss_race = self.process_batch(batch)
        self.log_dict({"train_loss_disease": loss_disease, "train_loss_sex": loss_sex, "train_loss_race": loss_race})
        wandb.log({"train_loss_disease": loss_disease, "train_loss_sex": loss_sex, "train_loss_race": loss_race})

        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)

        # if optimizer_idx == 0:
        #     return loss_disease
        # if optimizer_idx == 1:
        #     return loss_sex
        # if optimizer_idx == 2:
        #     return loss_race

    def validation_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race = self.process_batch(batch)
        self.log_dict({"val_loss_disease": loss_disease, "val_loss_sex": loss_sex, "val_loss_race": loss_race})

    def test_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race = self.process_batch(batch)
        self.log_dict({"test_loss_disease": loss_disease, "test_loss_sex": loss_sex, "test_loss_race": loss_race})


class DenseNet(pl.LightningModule):
    def __init__(self, num_classes_disease, num_classes_sex, num_classes_race, class_weights_race):
        super().__init__()
        self.automatic_optimization = False  # Disable automatic optimization
        self.num_classes_disease = num_classes_disease
        self.num_classes_sex = num_classes_sex
        self.num_classes_race = num_classes_race
        self.class_weights_race = torch.FloatTensor(class_weights_race)
        self.backbone = models.densenet121(pretrained=True)
        num_features = self.backbone.classifier.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes_disease)
        self.fc_sex = nn.Linear(num_features, self.num_classes_sex)
        self.fc_race = nn.Linear(num_features, self.num_classes_race)
        self.fc_connect = nn.Identity(num_features)
        self.backbone.classifier = self.fc_connect

        # Define accuracy and AUROC metrics for each task
        self.disease_accuracy = MulticlassAccuracy(num_classes=num_classes_disease)
        self.sex_accuracy = Accuracy(task='binary')
        self.race_accuracy = Accuracy(task='multiclass', num_classes=num_classes_race)
        
        self.disease_auroc = AUROC(task="multiclass", num_classes=num_classes_disease)
        self.sex_auroc = AUROC(task="multiclass", num_classes=num_classes_sex)
        self.race_auroc = AUROC(task="multiclass", num_classes=num_classes_race)
   
    def forward(self, x):
        embedding = self.backbone.forward(x)
        out_disease = self.fc_disease(embedding)
        out_sex = self.fc_sex(embedding)
        out_race = self.fc_race(embedding)
        return out_disease, out_sex, out_race

    def configure_optimizers(self):
        params_backbone = list(self.backbone.parameters())
        params_disease = params_backbone + list(self.fc_disease.parameters())
        params_sex = params_backbone + list(self.fc_sex.parameters())
        params_race = params_backbone + list(self.fc_race.parameters())
        optim_disease = torch.optim.Adam(params_disease, lr=0.001)
        optim_sex = torch.optim.Adam(params_sex, lr=0.001)
        optim_race = torch.optim.Adam(params_race, lr=0.001)

        return optim_disease, optim_sex, optim_race

    def unpack_batch(self, batch):
        return batch['image'], batch['label_disease'], batch['label_sex'], batch['label_race']

    def process_batch(self, batch):
        img, lab_disease, lab_sex, lab_race = self.unpack_batch(batch)
        out_disease, out_sex, out_race = self.forward(img)
        loss_disease = F.binary_cross_entropy(torch.sigmoid(out_disease), lab_disease)
        loss_sex = F.cross_entropy(out_sex, lab_sex)
        loss_race = F.cross_entropy(out_race, lab_race, weight=self.class_weights_race.type_as(img))
        
        # Calculate probabilities for AUROC calculation
        prob_disease = torch.sigmoid(out_disease)
        prob_sex = F.softmax(out_sex, dim=1)
        prob_race = F.softmax(out_race, dim=1)

        #return loss_disease, loss_sex, loss_race
        return loss_disease, loss_sex, loss_race, prob_disease, prob_sex, prob_race, lab_disease, lab_sex, lab_race

    # for multiple optimizers
    def training_step(self, batch, batch_idx): #, optimizer_idx):
        opt1, opt2, opt3 = self.optimizers()  # Manually access the optimizers
        #loss_disease, loss_sex, loss_race = self.process_batch(batch)
        loss_disease, loss_sex, loss_race, prob_disease, prob_sex, prob_race, lab_disease, lab_sex, lab_race = self.process_batch(batch)
        self.log_dict({"train_loss_disease": loss_disease, "train_loss_sex": loss_sex, "train_loss_race": loss_race})

        # Calculate and log accuracy for each task
        disease_acc = self.disease_accuracy(torch.argmax(prob_disease, dim=1), torch.argmax(lab_disease, dim=1))
        sex_acc = self.sex_accuracy((prob_sex[:, 1] > 0.5).int(), lab_sex)
        race_acc = self.race_accuracy(torch.argmax(prob_race, dim=1), lab_race)
        
        self.log_dict({"train_accuracy_disease": disease_acc, "train_accuracy_sex": sex_acc, "train_accuracy_race": race_acc})

        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)

        opt1.zero_grad()
        self.manual_backward(loss_disease, retain_graph=True)
        opt1.step()
        
        opt2.zero_grad()
        self.manual_backward(loss_sex, retain_graph=True)
        opt2.step()

        opt3.zero_grad()
        self.manual_backward(loss_race)
        opt3.step()
        
    def validation_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race, prob_disease, prob_sex, prob_race, lab_disease, lab_sex, lab_race = self.process_batch(batch)
        self.log_dict({"val_loss_disease": loss_disease, "val_loss_sex": loss_sex, "val_loss_race": loss_race})

        # Calculate and log accuracy for each task
        disease_acc = self.disease_accuracy(torch.argmax(prob_disease, dim=1), torch.argmax(lab_disease, dim=1))
        sex_acc = self.sex_accuracy((prob_sex[:, 1] > 0.5).int(), lab_sex)
        race_acc = self.race_accuracy(torch.argmax(prob_race, dim=1), lab_race)
        self.log_dict({"val_accuracy_disease": disease_acc, "val_accuracy_sex": sex_acc, "val_accuracy_race": race_acc})

        disease_auroc = self.disease_auroc(prob_disease, torch.argmax(lab_disease, dim=1))
        sex_auroc = self.sex_auroc(prob_sex, lab_sex)
        race_auroc = self.race_auroc(prob_race, lab_race)
        self.log_dict({"val_auroc_disease": disease_auroc, "val_auroc_sex": sex_auroc, "val_auroc_race": race_auroc})

    def test_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race, prob_disease, prob_sex, prob_race, lab_disease, lab_sex, lab_race = self.process_batch(batch)
        self.log_dict({"test_loss_disease": loss_disease, "test_loss_sex": loss_sex, "test_loss_race": loss_race})

        # Calculate and log accuracy for each task
        disease_acc = self.disease_accuracy(torch.argmax(prob_disease, dim=1), torch.argmax(lab_disease, dim=1))
        sex_acc = self.sex_accuracy((prob_sex[:, 1] > 0.5).int(), lab_sex)
        race_acc = self.race_accuracy(torch.argmax(prob_race, dim=1), torch.argmax(lab_race, dim=1))
        self.log_dict({"test_accuracy_disease": disease_acc, "test_accuracy_sex": sex_acc, "test_accuracy_race": race_acc})

        disease_auroc = self.disease_auroc(prob_disease, torch.argmax(lab_disease, dim=1))
        sex_auroc = self.sex_auroc(prob_sex, torch.argmax(lab_sex, dim=1))
        race_auroc = self.race_auroc(prob_race, torch.argmax(lab_race, dim=1))
        self.log_dict({"test_auroc_disease": disease_auroc, "test_auroc_sex": sex_auroc, "test_auroc_race": race_acc})


def test(model, data_loader, device):
    model.eval()
    logits_disease = []
    preds_disease = []
    targets_disease = []
    logits_sex = []
    preds_sex = []
    targets_sex = []
    logits_race = []
    preds_race = []
    targets_race = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab_disease, lab_sex, lab_race = batch['image'].to(device), batch['label_disease'].to(device), batch['label_sex'].to(device), batch['label_race'].to(device)
            out_disease, out_sex, out_race = model(img)

            pred_disease = torch.sigmoid(out_disease)
            pred_sex = torch.softmax(out_sex, dim=1)
            pred_race = torch.softmax(out_race, dim=1)

            logits_disease.append(out_disease)
            preds_disease.append(pred_disease)
            targets_disease.append(lab_disease)

            logits_sex.append(out_sex)
            preds_sex.append(pred_sex)
            targets_sex.append(lab_sex)

            logits_race.append(out_race)
            preds_race.append(pred_race)
            targets_race.append(lab_race)

        logits_disease = torch.cat(logits_disease, dim=0)
        preds_disease = torch.cat(preds_disease, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)

        logits_sex = torch.cat(logits_sex, dim=0)
        preds_sex = torch.cat(preds_sex, dim=0)
        targets_sex = torch.cat(targets_sex, dim=0)

        logits_race = torch.cat(logits_race, dim=0)
        preds_race = torch.cat(preds_race, dim=0)
        targets_race = torch.cat(targets_race, dim=0)

        counts = []
        for i in range(0,num_classes_disease):
            t = targets_disease[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

        counts = []
        for i in range(0,num_classes_sex):
            t = targets_sex == i
            c = torch.sum(t)
            counts.append(c)
        print(counts)

        counts = []
        for i in range(0,num_classes_race):
            t = targets_race == i
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds_disease.cpu().numpy(), targets_disease.cpu().numpy(), logits_disease.cpu().numpy(), preds_sex.cpu().numpy(), targets_sex.cpu().numpy(), logits_sex.cpu().numpy(), preds_race.cpu().numpy(), targets_race.cpu().numpy(), logits_race.cpu().numpy()


def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets_disease = []
    targets_sex = []
    targets_race = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab_disease, lab_sex, lab_race = batch['image'].to(device), batch['label_disease'].to(device), batch['label_sex'].to(device), batch['label_race'].to(device)
            emb = model.backbone(img)
            embeds.append(emb)
            targets_disease.append(lab_disease)
            targets_sex.append(lab_sex)
            targets_race.append(lab_race)

        embeds = torch.cat(embeds, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)
        targets_sex = torch.cat(targets_sex, dim=0)
        targets_race = torch.cat(targets_race, dim=0)

    return embeds.cpu().numpy(), targets_disease.cpu().numpy(), targets_sex.cpu().numpy(), targets_race.cpu().numpy()


def main(hparams):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    if hparams.sample == True:
        sample = '_subset'
    else:
        sample = ''
    # data
    data = CheXpertDataModule(csv_train_img=f'../datafiles/chexpert/chexpert.sample.train{sample}.csv',
                              csv_val_img=f'../datafiles/chexpert/chexpert.sample.val{sample}.csv',
                              csv_test_img=f'../datafiles/chexpert/chexpert.sample.test{sample}.csv',
                              image_size=image_size,
                              pseudo_rgb=True,
                              batch_size=batch_size,
                              num_workers=num_workers)

    # model
    model_type = DenseNet
    model = model_type(num_classes_disease=num_classes_disease, num_classes_sex=num_classes_sex, num_classes_race=num_classes_race, class_weights_race=class_weights_race)

    # Create output directory
    out_name = 'densenet-all'
    out_dir = 'chexpert/multitask/' + out_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image'].astype(np.uint8))

    wandb_logger = WandbLogger()
    #checkpoint_callback = ModelCheckpoint(monitor="val_loss_disease", mode='min')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss_disease", 
                                        mode='min', 
                                        dirpath='models/',
                                        filename= 'chexpert-multitask-densenet-{epoch:02d}-{val_loss_disease:.2f}',
                                        save_top_k=3,        # Number of best checkpoints to keep
                                        save_last=True,
                                        verbose=True)
    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 5,
        max_epochs=epochs,
        devices=hparams.gpus,
        logger=wandb_logger 
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    print('Best Model Path', trainer.checkpoint_callback.best_model_path)
    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes_disease=num_classes_disease, num_classes_sex=num_classes_sex, num_classes_race=num_classes_race, class_weights_race=class_weights_race)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")

    model.to(device)

    cols_names_classes_disease = ['class_' + str(i) for i in range(0,num_classes_disease)]
    cols_names_logits_disease = ['logit_' + str(i) for i in range(0, num_classes_disease)]
    cols_names_targets_disease = ['target_' + str(i) for i in range(0, num_classes_disease)]

    cols_names_classes_sex = ['class_' + str(i) for i in range(0,num_classes_sex)]
    cols_names_logits_sex = ['logit_' + str(i) for i in range(0, num_classes_sex)]

    cols_names_classes_race = ['class_' + str(i) for i in range(0,num_classes_race)]
    cols_names_logits_race = ['logit_' + str(i) for i in range(0, num_classes_race)]

    print('VALIDATION')
    preds_val_disease, targets_val_disease, logits_val_disease, preds_val_sex, targets_val_sex, logits_val_sex, preds_val_race, targets_val_race, logits_val_race = test(model, data.val_dataloader(), device)
    
    df = pd.DataFrame(data=preds_val_disease, columns=cols_names_classes_disease)
    df['Model Path'] = trainer.checkpoint_callback.best_model_path
    df_logits = pd.DataFrame(data=logits_val_disease, columns=cols_names_logits_disease)
    df_targets = pd.DataFrame(data=targets_val_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    df.to_csv(os.path.join(out_dir, f'predictions.val.disease.{date}.csv'), index=False)

    df = pd.DataFrame(data=preds_val_sex, columns=cols_names_classes_sex)
    df['Model Path'] = trainer.checkpoint_callback.best_model_path
    df_logits = pd.DataFrame(data=logits_val_sex, columns=cols_names_logits_sex)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_val_sex
    df.to_csv(os.path.join(out_dir, f'predictions.val.sex.{date}.csv'), index=False)

    df = pd.DataFrame(data=preds_val_race, columns=cols_names_classes_race)
    df['Model Path'] = trainer.checkpoint_callback.best_model_path
    df_logits = pd.DataFrame(data=logits_val_race, columns=cols_names_logits_race)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_val_race
    df.to_csv(os.path.join(out_dir, f'predictions.val.race.{date}.csv'), index=False)

    print('TESTING')
    preds_test_disease, targets_test_disease, logits_test_disease, preds_test_sex, targets_test_sex, logits_test_sex, preds_test_race, targets_test_race, logits_test_race = test(model, data.test_dataloader(), device)
    
    df = pd.DataFrame(data=preds_test_disease, columns=cols_names_classes_disease)
    df['Model Path'] = trainer.checkpoint_callback.best_model_path
    df_logits = pd.DataFrame(data=logits_test_disease, columns=cols_names_logits_disease)
    df_targets = pd.DataFrame(data=targets_test_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, f'predictions.test.disease.{date}.csv'), index=False)

    df = pd.DataFrame(data=preds_test_sex, columns=cols_names_classes_sex)
    df['Model Path'] = trainer.checkpoint_callback.best_model_path
    df_logits = pd.DataFrame(data=logits_test_sex, columns=cols_names_logits_sex)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_test_sex
    df.to_csv(os.path.join(out_dir, f'predictions.test.sex.{date}.csv'), index=False)

    df = pd.DataFrame(data=preds_test_race, columns=cols_names_classes_race)
    df['Model Path'] = trainer.checkpoint_callback.best_model_path
    df_logits = pd.DataFrame(data=logits_test_race, columns=cols_names_logits_race)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_test_race
    df.to_csv(os.path.join(out_dir, f'predictions.test.race.{date}.csv'), index=False)

    wandb.finish()

    print('EMBEDDINGS')

    embeds_val, targets_val_disease, targets_val_sex, targets_val_race = embeddings(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=embeds_val)
    df['Model Path'] = trainer.checkpoint_callback.best_model_path
    df_targets_disease = pd.DataFrame(data=targets_val_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_targets_disease], axis=1)
    df['target_sex'] = targets_val_sex
    df['target_race'] = targets_val_race
    df.to_csv(os.path.join(out_dir, f'embeddings.val.{date}.csv'), index=False)

    embeds_test, targets_test_disease, targets_test_sex, targets_test_race = embeddings(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=embeds_test)
    df['Model Path'] = trainer.checkpoint_callback.best_model_path
    df_targets_disease = pd.DataFrame(data=targets_test_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_targets_disease], axis=1)
    df['target_sex'] = targets_test_sex
    df['target_race'] = targets_test_race
    df.to_csv(os.path.join(out_dir, f'embeddings.test.{date}.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    parser.add_argument('--wb', default=True)
    parser.add_argument('--sample', default=True)

    args = parser.parse_args()
    print(args)

    if args.wb == True:
        wandb.init(
        # set the wandb project where this run will be logged
        project="cheXploration",
        # track hyperparameters and run metadata
        config={
        "Architecture": "DenseNet121",
        "Dataset": "CheXpert",
        "Test": args.sample,
        "Target": "Multitask",
        }
        )
    main(args)

