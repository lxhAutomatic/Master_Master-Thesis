# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 18:50:25 2023

@author: Xinhao Lan
"""
import matplotlib 
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow.keras.backend as kb
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import shutil
import json
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import models

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
chexpert_train_csv = pd.read_csv('/home/bme001/20225898/CheXpert-v1.0-small/train.csv', sep = ',')
chexpert_valid_csv = pd.read_csv('/home/bme001/20225898/CheXpert-v1.0-small/valid.csv', sep = ',')
chexpert_test_csv = pd.read_csv('/home/bme001/20225898/CheXpert-v1.0-small/test.csv', sep = ',')
chexpert_train_csv = chexpert_train_csv.fillna(1)
chexpert_valid_csv = chexpert_valid_csv.fillna(1)
chexpert_test_csv = chexpert_test_csv.fillna(1)
chexpert_train_csv = chexpert_train_csv.replace(-1, 1)
chexpert_valid_csv = chexpert_valid_csv.replace(-1, 1)
chexpert_test_csv = chexpert_test_csv.replace(-1, 1)
chexpert_train_df = chexpert_train_csv[['Path', 'Pneumothorax', 'Pleural Effusion']]
chexpert_valid_df = chexpert_valid_csv[['Path', 'Pneumothorax', 'Pleural Effusion']]
chexpert_test_df = chexpert_test_csv[['Path', 'Pneumothorax', 'Pleural Effusion']]
output_dir = '/home/bme001/20225898/baseline_result_1/'

all_pathologies = ['Pneumothorax']

base_generator = ImageDataGenerator(rescale = 1./255)
def flow_from_dataframe(image_generator, dataframe, batch_size):
    df_gen = image_generator.flow_from_dataframe(dataframe, x_col = 'Path', y_col = all_pathologies, target_size = (389, 320), classes = all_pathologies,
                                                 color_mode = 'rgb', class_mode = 'raw', shuffle = False, batch_size = batch_size)
    return df_gen

train_gen = flow_from_dataframe(image_generator=base_generator, dataframe= chexpert_train_df, batch_size = 16)
valid_gen = flow_from_dataframe(image_generator=base_generator, dataframe= chexpert_valid_df, batch_size = 16)
test_gen = flow_from_dataframe(image_generator=base_generator, dataframe= chexpert_test_df, batch_size = 16)

train_x, train_y = next(train_gen)

# print('The dimension of the image is:', train_x[1].shape)
# print('The dimension of the illness labe is:',len(train_y[1]))

# Models

input_shape = (389, 320, 3)
img_input = Input(shape = input_shape)
base_model = DenseNet121(include_top=False, input_tensor=img_input, input_shape=input_shape, pooling="max", weights='imagenet')
x = base_model.output
predictions = Dense(len(all_pathologies), activation="sigmoid", name="predictions")(x)
model = Model(inputs=img_input, outputs=predictions)

# Train the model

model_train = model
output_weights_name = 'weight_1.h5'
checkpoint = ModelCheckpoint(output_weights_name, save_weights_only = True,
                             save_best_only = True, verbose = 1)


class MultipleClassAUROC(Callback):
    def __init__(self, generator, class_names, weights_path, stats = None):
        super(Callback, self).__init__()
        self.generator = generator
        self.class_names = class_names
        self.weights_path = weights_path
        self.best_weights_path = os.path.join(os.path.split(weights_path)[0], 
                                              f"best_{os.path.split(weights_path)[1]}",)
        self.best_auroc_log_path = os.path.join(os.path.split(weights_path)[0],
                                                "best_auroc.log")
        self.stats_output_path = os.path.join(os.path.split(weights_path)[0],
                                              ".training_stats.json")
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}
        
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []
    def on_epoch_end(self, epoch, logs = {}):
        print("\n*********************************")
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        print(f"Learning Rate actual: {self.stats['lr']}")
        y_hat = self.model.predict_generator(self.generator,steps=self.generator.n/self.generator.batch_size)
        y = self.generator.labels
        print(f"*** epoch#{epoch + 1} ROC Curves Training Phase ***")
        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]].append(score)
            current_auroc.append(score)
            print(f"{i+1}. {self.class_names[i]}: {score}")
        print("*********************************")
        
        mean_auroc = np.mean(current_auroc)
        print(f"Mean ROC curves: {mean_auroc}")
        if mean_auroc > self.stats["best_mean_auroc"]:
            print(f"Update the result of ROC from {self.stats['best_mean_auroc']} to {mean_auroc}")
            shutil.copy(self.weights_path, self.best_weights_path)
            print(f"Update the log files: {self.best_auroc_log_path}")
            with open(self.best_auroc_log_path, "a") as f:
                f.write(f"(epoch#{epoch + 1}) auroc: {mean_auroc}, lr:{self.stats['lr']}\n")
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)
            print(f"Update the weights: {self.weights_path} -> {self.best_weights_path}")
            self.stats["best_mean_auroc"] = mean_auroc
            print("*********************************")
        return
                            
training_stats = {}
auroc = MultipleClassAUROC(
    generator = valid_gen,
    class_names = all_pathologies,
    weights_path = output_weights_name,
    stats = training_stats)


initial_learning_rate = 1e-2
optimizer = Adam(learning_rate = initial_learning_rate)
model_train.compile(optimizer = optimizer, loss="binary_crossentropy")

logs_base_dir = output_dir + 'working/' #set the dir
patience_reduce_lr = 2
min_lr = 1e-4
callbacks = [checkpoint, 
             TensorBoard(log_dir=os.path.join(logs_base_dir,"logs"),batch_size=train_gen.batch_size),
             ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = patience_reduce_lr, verbose=1,mode="min",min_lr=min_lr),
             auroc,]

epochs = 10
fit_history = model.fit_generator(
    generator = train_gen,
    steps_per_epoch=train_gen.n/train_gen.batch_size,
    epochs=epochs,
    validation_data=valid_gen,
    validation_steps=valid_gen.n/valid_gen.batch_size,
    callbacks=callbacks,
    shuffle=False
    )

plt.figure(1, figsize = (15,8))
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])
plt.savefig(output_dir + 'loss.png', dpi = 600)

model.load_weights('/home/bme001/20225898/weight_1.h5')
test_gen.reset()
pred_y = model.predict_generator(test_gen, steps = test_gen.n/test_gen.batch_size,verbose=True)
test_gen.reset()
test_x, test_y = next(test_gen)
np.savetxt(output_dir + 'pred_y.txt', pred_y, fmt = '%f')
np.savetxt(output_dir + 'test_y.txt', test_y, fmt = '%f')

model.load_weights('/home/bme001/20225898/weight_1.h5')
valid_y = model.predict_generator(valid_gen, steps = valid_gen.n/valid_gen.batch_size,verbose=True)
thresholds_list_valid = []
for (idx, c_label) in enumerate(all_pathologies):
    fpr, tpr, thresholds = roc_curve(valid_gen.labels[:,idx].astype(int), valid_y[:,idx])
    thresholds_list_valid.append(thresholds)
np.savetxt(output_dir + 'validation_thresholds_list.txt', thresholds_list_valid, fmt = '%f')

fpr_list = []
tpr_list = []
thresholds_list = []
fig, c_ax = plt.subplots(1,1,figsize=(9,9))
for (idx, c_label) in enumerate(all_pathologies):
    fpr, tpr, thresholds = roc_curve(test_gen.labels[:,idx].astype(int), pred_y[:,idx])
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    thresholds_list.append(thresholds)
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig(output_dir + 'barely_trained_net.png')
auc = roc_auc_score(test_gen.labels, pred_y)
print('ROC AUC: %f' % auc)

np.savetxt(output_dir + 'fpr_list.txt', fpr_list, fmt = '%f')
np.savetxt(output_dir + 'tpr_list.txt', tpr_list, fmt = '%f')
np.savetxt(output_dir + 'thresholds_list.txt', thresholds_list, fmt = '%f')
np.savetxt(output_dir + 'true_labels.txt', test_gen.labels,fmt='%f')
np.savetxt(output_dir + 'pred_labels.txt', pred_y,fmt='%f')















