
from datetime import datetime
import glob
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
from PIL import Image
import random as python_random
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import auc, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbCallback

# pip install image-classifiers==1.0.0b1
from classification_models.tfkeras import Classifiers
# More information about this package can be found at https://github.com/qubvel/classification_models
# Check if a GPU is available
tf.keras.backend.clear_session()

if tf.config.list_physical_devices('GPU'):
    print("GPU available")
    # Set TensorFlow to use the GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

np.random.seed(2021)
python_random.seed(2021)
tf.random.set_seed(2021)

wandb.init(
                # set the wandb project where this run will be logged
                project="chexnet-" + 'race' + "-pred",
                # track hyperparameters and run metadata
                config={"architecture": "ResNet34",
                        "dataset": "CheXpert",
                        "fine-tuned": True, 
                        "target": 'race', 
                        "package": 'tensorflow',
                        "data-subset": False
                }
                )

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# CheXpert images can be found: https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
data_df = pd.read_csv('train_cheXbert.csv')

# Demographic labels can be found: https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf
demo_df = pd.DataFrame(pd.read_excel("CHEXPERT_DEMO.xlsx", engine='openpyxl')) #pip install openpyxl

# 60-10-30, train-val-test split that we used
# These splits can be found in this repository
split_df = pd.read_csv('chexpert_split_2021_08_20.csv').set_index('index')

# All preprocessing steps of CheXpert .jpg images are included in this repository
# Image data preprocessing include resizing to 320x320
# and normalizing images with ImageNet mean and standard deviation values
# using resnet34, preprocess_input = Classifiers.get('resnet34') from the classification_models.tfkeras package

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
data_df[:0]
len(data_df)
data_df.split.value_counts(normalize=True)
data_df.race.value_counts(normalize=True)
data_df[['split', 'race']].value_counts(normalize=True)

train_df = data_df[data_df.split=="train"]
validation_df = data_df[data_df.split=="validate"]
test_df = data_df[data_df.split=="test"]

train_df.to_csv('train_df.csv')
validation_df.to_csv('validation_df.csv')
test_df.to_csv('test_df.csv')

unique_train_id = train_df.patient_id.unique()
unique_validation_id = validation_df.patient_id.unique()
unique_test_id = test_df.patient_id.unique()
all_id = np.concatenate((unique_train_id, unique_validation_id, unique_test_id), axis=None)

def contains_duplicates(X):
    return len(np.unique(X)) != len(X)

print(contains_duplicates(all_id))

HEIGHT, WIDTH = 320, 320

arc_name = "CHEXPERT_RACE_RESNET34_TF" 
resnet34, preprocess_input = Classifiers.get('resnet34')

input_a = Input(shape=(HEIGHT, WIDTH, 3))
base_model = resnet34(input_tensor=input_a, include_top=False, input_shape=(HEIGHT,WIDTH,3), weights='imagenet')
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(3, name='dense_logits')(x)
output = Activation('softmax', dtype='float32', name='predictions')(x)
model = Model(inputs=[input_a], outputs=[output])

learning_rate = 1e-3
momentum_val=0.9
decay_val= 0.0
train_batch_size = 256 # may need to reduce batch size if OOM error occurs
test_batch_size = 256
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=2, min_lr=1e-5, verbose=1)

adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay_val)
adam_opt = tf.keras.mixed_precision.LossScaleOptimizer(adam_opt)

model.compile(optimizer=adam_opt,
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC(curve='ROC', name='ROC-AUC'),
                    tf.keras.metrics.AUC(curve='PR', name='PR-AUC')
                ],
)

train_gen = ImageDataGenerator(
            rotation_range=15,
            fill_mode='constant',
            horizontal_flip=True,
            zoom_range=0.1,
            preprocessing_function=preprocess_input
            )

validate_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_batches = train_gen.flow_from_dataframe(train_df, directory="../", x_col="Path", y_col="race", class_mode="categorical",target_size=(HEIGHT, WIDTH),shuffle=True,seed=2021,batch_size=train_batch_size, dtype='float32')
validate_batches = validate_gen.flow_from_dataframe(validation_df, directory="../", x_col="Path", y_col="race", class_mode="categorical",target_size=(HEIGHT, WIDTH),shuffle=False,batch_size=test_batch_size, dtype='float32')        

train_epoch = math.ceil(len(train_df) / train_batch_size)
val_epoch = math.ceil(len(validation_df) / test_batch_size)
print(train_epoch, val_epoch)

var_date = datetime.now().strftime("%Y%m%d-%H%M%S")
ES = EarlyStopping(monitor='val_loss', mode='min', patience=4, restore_best_weights=True)
checkloss = ModelCheckpoint("models/" + str(arc_name) + "_" + var_date+"_epoch:{epoch:03d}_val_loss:{val_loss:.2f}.hdf5", monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=False)

model.fit(train_batches,
            validation_data=validate_batches,
            epochs=100,
            steps_per_epoch=int(train_epoch),
            validation_steps=int(val_epoch),
            workers=32,
            max_queue_size=50,
            shuffle=False,
            callbacks=[checkloss, reduce_lr, ES, WandbCallback()]
           )

# test_batches = validate_gen.flow_from_dataframe(test_stratified, directory="../", x_col="Path", y_col="race", class_mode="categorical",target_size=(HEIGHT, WIDTH),shuffle=False,batch_size=test_batch_size, dtype='float32')        

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