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

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torchmetrics.classification import MulticlassAccuracy, AUROC
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser

image_size = (224, 224)
num_classes = 14
batch_size = 150
epochs = 20
num_workers = 4
img_data_dir = '/n/groups/patel/aashna/xray-dl/data/'

torch.cuda.empty_cache()

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

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = img_data_dir + self.data.loc[idx, 'path_preproc']
            img_label = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label[i] = np.array(self.data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            sample = {'image_path': img_path, 'label': img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image']).unsqueeze(0)
        label = torch.from_numpy(sample['label'])

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        image = imread(sample['image_path']).astype(np.float32)

        return {'image': image, 'label': sample['label']}


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
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.resnet34(pretrained=True)
        # freeze_model(self.model)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

    def remove_head(self):
        num_features = self.model.fc.in_features
        id_layer = nn.Identity(num_features)
        self.model.fc = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss, on_epoch=True)
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        #self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


class DenseNet(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = models.densenet121(pretrained=True)
        # freeze_model(self.model)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)

    def remove_head(self):
        num_features = self.model.classifier.in_features
        id_layer = nn.Identity(num_features)
        self.model.classifier = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer
    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return prob, lab, loss

    def training_step(self, batch, batch_idx):
        prob, lab, loss = self.process_batch(batch)
        
        # Calculate accuracy
        predictions = torch.argmax(prob, dim=1)
        labels = torch.argmax(lab, dim=1)
        train_acc = self.train_accuracy(predictions, labels)
        self.log_dict({'train_loss': loss, 'train_accuracy': train_acc}, on_epoch=True)

        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        #self.logger.experiment.add_image('images', grid, self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        prob, lab, loss = self.process_batch(batch)

        # Calculate accuracy
        predictions = torch.argmax(prob, dim=1)
        labels = torch.argmax(lab, dim=1)
        val_acc = self.val_accuracy(predictions, labels)
        val_auroc = self.val_auroc(prob, labels)
        #print('Validation Loss: ', loss, 'Validation AUROC:', val_auroc, 'Validation Accuracy:', val_acc)
        self.log_dict({'val_loss': loss, 'val_accuracy': val_acc, 'val_auroc': val_auroc}, on_epoch=True)


    def test_step(self, batch, batch_idx):
        prob, lab, loss = self.process_batch(batch)

        # Calculate accuracy
        predictions = torch.argmax(prob, dim=1)
        labels = torch.argmax(lab, dim=1)
        test_acc = self.test_accuracy(predictions, labels)
        test_auroc = self.test_auroc(prob, labels)
        self.log_dict({'test_loss': loss, 'test_accuracy': test_acc, 'test_auroc': test_auroc}, on_epoch=True)
                  
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def test(model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0,num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()


def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            emb = model(img)
            embeds.append(emb)
            targets.append(lab)

        embeds = torch.cat(embeds, dim=0)
        targets = torch.cat(targets, dim=0)

    return embeds.cpu().numpy(), targets.cpu().numpy()


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
    model = model_type(num_classes=num_classes)

    # Create output directory
    out_name = 'densenet-all'
    out_dir = 'chexpert/disease/' + out_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    temp_dir = os.path.join(out_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.jpg'), sample['image']) #.astype(np.uint8))

    wandb_logger = WandbLogger()
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                        mode='min', 
                                        dirpath='models/',
                                        filename= 'chexpert-disease-densenet-{epoch:02d}-{val_loss:.2f}',
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

    print('Model Path:', trainer.checkpoint_callback.best_model_path)
    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes=num_classes)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")

    model.to(device)

    cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

    print('VALIDATION')
    preds_val, targets_val, logits_val = test(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
    df['Model Path'] = trainer.checkpoint_callback.best_model_path
    df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.val.csv'), index=False)

    print('TESTING')
    preds_test, targets_test, logits_test = test(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df['Model Path'] = trainer.checkpoint_callback.best_model_path
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.test.csv'), index=False)

    wandb.finish()

    print('EMBEDDINGS')

    model.remove_head()

    embeds_val, targets_val = embeddings(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=embeds_val)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'embeddings.val.csv'), index=False)

    embeds_test, targets_test = embeddings(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=embeds_test)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'embeddings.test.csv'), index=False)


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
        "Target": "Disease",
        "Image Size": "224x224"
        } 
        )

    main(args)
