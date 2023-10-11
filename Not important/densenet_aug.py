# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:30:58 2023

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
import shutil
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve, auc
import os

chexpert_train_csv = pd.read_csv('/home/bme001/20225898/CheXpert-v1.0-small/train.csv', sep = ',')
chexpert_valid_csv = pd.read_csv('/home/bme001/20225898/CheXpert-v1.0-small/valid.csv', sep = ',')
chexpert_test_csv = pd.read_csv('/home/bme001/20225898/CheXpert-v1.0-small/test.csv', sep = ',')

chexpert_train_csv = chexpert_train_csv.fillna(0)
chexpert_valid_csv = chexpert_valid_csv.fillna(0)
chexpert_test_csv = chexpert_test_csv.fillna(0)

chexpert_train_csv = chexpert_train_csv.replace(-1, 1)
chexpert_valid_csv = chexpert_valid_csv.replace(-1, 1)
chexpert_test_csv = chexpert_test_csv.replace(-1, 1)

chexpert_train_df = chexpert_train_csv[['Path', 'Pneumothorax', 'Pleural Effusion']]
chexpert_valid_df = chexpert_valid_csv[['Path', 'Pneumothorax', 'Pleural Effusion']]
chexpert_test_df = chexpert_test_csv[['Path', 'Pneumothorax', 'Pleural Effusion']]

#chexpert_train_df_1, chexpert_test_df_1 = train_test_split(chexpert_train_df, test_size = 0.2, random_state = 0, stratify = chexpert_train_df['Pneumothorax'])
#chexpert_train_df_2, chexpert_test_df_2 = train_test_split(chexpert_train_df, test_size = 0.2, random_state = 0, stratify = chexpert_train_df['Pleural Effusion'])
output_dir_1 = '/home/bme001/20225898/baseline_result_4/'
#output_dir_2 = '/home/bme001/20225898/baseline_result_2/'
all_pathologies_1 = ['Pneumothorax']
#all_pathologies_2 = ['Pleural Effusion']

#依照label将train的集拆成0和1
chexpert_train_df_0 = chexpert_train_df[chexpert_train_df['Pneumothorax'].isin([0])]
chexpert_train_df_1 = chexpert_train_df[chexpert_train_df['Pneumothorax'].isin([1])]

subset_0 = chexpert_train_df_0.sample(frac=0.1125)
chexpert_train_df = pd.concat([subset_0, chexpert_train_df_1])



def mainbody (chexpert_train_df, chexpert_valid_df, chexpert_test_df, all_pathologies, output_dir):
    base_generator = ImageDataGenerator(rescale = 1./255, rotation_range = 20, width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.2,horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
    def flow_from_dataframe(image_generator, dataframe, batch_size):
        df_gen = image_generator.flow_from_dataframe(dataframe, x_col = 'Path', y_col = all_pathologies, target_size = (389, 320), classes = all_pathologies,
                                                     color_mode = 'rgb', class_mode = 'raw', shuffle = False, batch_size = batch_size)
        return df_gen

    train_gen = flow_from_dataframe(image_generator=base_generator, dataframe= chexpert_train_df, batch_size = 16)
    valid_gen = flow_from_dataframe(image_generator=base_generator, dataframe= chexpert_valid_df, batch_size = 16)
    test_gen = flow_from_dataframe(image_generator=base_generator, dataframe= chexpert_test_df, batch_size = 16)
    
    train_x, train_y = next(train_gen)
    input_shape = (389, 320, 3)
    img_input = Input(shape = input_shape)
    base_model = DenseNet121(include_top=False, input_tensor=img_input, input_shape=input_shape, pooling="max", weights='imagenet')
    x = base_model.output
    predictions = Dense(len(all_pathologies), activation="sigmoid", name="predictions")(x)
    model = Model(inputs=img_input, outputs=predictions)
    model_train = model
    output_weights_name = 'weight.h5'
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
    """
    fit_history = model.fit_generator(
        generator = train_gen,
        steps_per_epoch=train_gen.n/train_gen.batch_size,
        epochs=epochs,
        validation_data=valid_gen,
        validation_steps=valid_gen.n/valid_gen.batch_size,
        callbacks=callbacks,
        shuffle=False
        )
    """
    fit_history = model.fit_generator(
        generator = train_gen,
        steps_per_epoch=1000,
        epochs=50,
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
    
    model.load_weights('/home/bme001/20225898/weight.h5')
    pred_y = model.predict_generator(test_gen, steps = test_gen.n/test_gen.batch_size,verbose=True)
    test_gen.reset()
    test_x, test_y = next(test_gen)
    seed = np.random.randint(16, size=1)
    print(f"Image index {seed}")
    print(f"Vector of the pathologies label: {test_y[seed[0]]}")
    print(f"Vector of the predicted pathologies label: {pred_y[seed[0]]}")

    model.load_weights('/home/bme001/20225898/weight.h5')
    thresholds_list_valid = []
    for (idx, c_label) in enumerate(all_pathologies):
        fpr, tpr, thresholds = roc_curve(test_gen.labels[:,idx].astype(int), pred_y[:,idx])
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
    auc_1 = roc_auc_score(test_gen.labels, pred_y)
    print('ROC AUC: %f' % auc_1)
    np.savetxt(output_dir + 'fpr_list.txt', fpr_list, fmt = '%f')
    np.savetxt(output_dir + 'tpr_list.txt', tpr_list, fmt = '%f')
    np.savetxt(output_dir + 'thresholds_list.txt', thresholds_list, fmt = '%f')
    np.savetxt(output_dir + 'true_labels.txt',test_gen.labels,fmt='%f')
    np.savetxt(output_dir + 'pred_labels.txt',pred_y,fmt='%f')
mainbody(chexpert_train_df, chexpert_valid_df, chexpert_test_df, all_pathologies_1, output_dir_1)
#mainbody(chexpert_train_df_2, chexpert_valid_df, chexpert_test_df, all_pathologies_2, output_dir_2)