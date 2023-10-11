# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:25:12 2023

@author: Xinhao Lan
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

path = 'D:/CheXpert-v1.0-small/'
chexpert_train_csv = pd.read_csv(path + 'train.csv', sep = ',')
chexpert_valid_csv = pd.read_csv(path + 'valid.csv', sep = ',')
chexpert_train_fit = chexpert_train_csv.fillna(0)
chexpert_valid_fit = chexpert_valid_csv.fillna(0)
chexpert_train_fit = chexpert_train_fit.replace(-1, 1)
chexpert_valid_df = chexpert_valid_fit.replace(-1, 1)
chexpert_train_df, chexpert_test_df = train_test_split(chexpert_train_fit, test_size = 0.2, random_state = 0, stratify = chexpert_train_fit['Pneumothorax'])
with open("D:/CheXNet/train_list.txt","w") as file:
    for i in range(chexpert_train_df.shape[0]):
        temp_rows = (chexpert_train_df.iloc[i]).copy()
        oldpath = 'D:/' + temp_rows['Path']
        if temp_rows['Path'][45] == '/':
            newpath = temp_rows['Path'][26:38] + '_' + temp_rows['Path'][39:45] + '_' + temp_rows['Path'][46:]
        else:
            newpath = temp_rows['Path'][26:38] + '_' + temp_rows['Path'][39:46] + '_' + temp_rows['Path'][47:]
        print('image', i, newpath)
        shutil.copy(oldpath, 'D:/CheXNet/images/' + newpath)
        image_name = newpath + ' '
        label_1 = str(int(temp_rows['No Finding'])) + ' '
        label_2 = str(int(temp_rows['Enlarged Cardiomediastinum'])) + ' '
        label_3 = str(int(temp_rows['Cardiomegaly'])) + ' '
        label_4 = str(int(temp_rows['Lung Opacity'])) + ' '
        label_5 = str(int(temp_rows['Lung Lesion'])) + ' '
        label_6 = str(int(temp_rows['Edema'])) + ' '
        label_7 = str(int(temp_rows['Consolidation'])) + ' '
        label_8 = str(int(temp_rows['Pneumonia'])) + ' '
        label_9 = str(int(temp_rows['Atelectasis'])) + ' '
        label_10 = str(int(temp_rows['Pneumothorax'])) + ' '
        label_11 = str(int(temp_rows['Pleural Effusion'])) + ' '
        label_12 = str(int(temp_rows['Pleural Other'])) + ' '
        label_13 = str(int(temp_rows['Fracture'])) + ' '
        label_14 = str(int(temp_rows['Support Devices'])) + ' '
        file.write(image_name + label_1 + label_2 + label_3 + label_4 + label_5 + label_6 + label_7 + label_8 + label_9 + 
                   label_10 + label_11 + label_12 + label_13 + label_14)
        file.write('\n')
with open("D:/CheXNet/test_list.txt","w") as file:
    for i in range(chexpert_test_df.shape[0]):
        temp_rows = (chexpert_test_df.iloc[i]).copy()
        oldpath = 'D:/' + temp_rows['Path']
        if temp_rows['Path'][45] == '/':
            newpath = temp_rows['Path'][26:38] + '_' + temp_rows['Path'][39:45] + '_' + temp_rows['Path'][46:]
        else:
            newpath = temp_rows['Path'][26:38] + '_' + temp_rows['Path'][39:46] + '_' + temp_rows['Path'][47:]
        print('image', i, newpath)
        shutil.copy(oldpath, 'D:/CheXNet/images/' + newpath)
        image_name = newpath + ' '
        label_1 = str(int(temp_rows['No Finding'])) + ' '
        label_2 = str(int(temp_rows['Enlarged Cardiomediastinum'])) + ' '
        label_3 = str(int(temp_rows['Cardiomegaly'])) + ' '
        label_4 = str(int(temp_rows['Lung Opacity'])) + ' '
        label_5 = str(int(temp_rows['Lung Lesion'])) + ' '
        label_6 = str(int(temp_rows['Edema'])) + ' '
        label_7 = str(int(temp_rows['Consolidation'])) + ' '
        label_8 = str(int(temp_rows['Pneumonia'])) + ' '
        label_9 = str(int(temp_rows['Atelectasis'])) + ' '
        label_10 = str(int(temp_rows['Pneumothorax'])) + ' '
        label_11 = str(int(temp_rows['Pleural Effusion'])) + ' '
        label_12 = str(int(temp_rows['Pleural Other'])) + ' '
        label_13 = str(int(temp_rows['Fracture'])) + ' '
        label_14 = str(int(temp_rows['Support Devices'])) + ' '
        file.write(image_name + label_1 + label_2 + label_3 + label_4 + label_5 + label_6 + label_7 + label_8 + label_9 + 
                   label_10 + label_11 + label_12 + label_13 + label_14)
        file.write('\n')
with open("D:/CheXNet/val_list.txt","w") as file:
    for i in range(chexpert_valid_df.shape[0]):
        temp_rows = (chexpert_valid_df.iloc[i]).copy()
        oldpath = 'D:/' + temp_rows['Path']
        if temp_rows['Path'][45] == '/':
            newpath = temp_rows['Path'][26:38] + '_' + temp_rows['Path'][39:45] + '_' + temp_rows['Path'][46:]
        else:
            newpath = temp_rows['Path'][26:38] + '_' + temp_rows['Path'][39:46] + '_' + temp_rows['Path'][47:]
        print('image', i, newpath)
        shutil.copy(oldpath, 'D:/CheXNet/images/' + newpath)
        image_name = newpath + ' '
        label_1 = str(int(temp_rows['No Finding'])) + ' '
        label_2 = str(int(temp_rows['Enlarged Cardiomediastinum'])) + ' '
        label_3 = str(int(temp_rows['Cardiomegaly'])) + ' '
        label_4 = str(int(temp_rows['Lung Opacity'])) + ' '
        label_5 = str(int(temp_rows['Lung Lesion'])) + ' '
        label_6 = str(int(temp_rows['Edema'])) + ' '
        label_7 = str(int(temp_rows['Consolidation'])) + ' '
        label_8 = str(int(temp_rows['Pneumonia'])) + ' '
        label_9 = str(int(temp_rows['Atelectasis'])) + ' '
        label_10 = str(int(temp_rows['Pneumothorax'])) + ' '
        label_11 = str(int(temp_rows['Pleural Effusion'])) + ' '
        label_12 = str(int(temp_rows['Pleural Other'])) + ' '
        label_13 = str(int(temp_rows['Fracture'])) + ' '
        label_14 = str(int(temp_rows['Support Devices'])) + ' '
        file.write(image_name + label_1 + label_2 + label_3 + label_4 + label_5 + label_6 + label_7 + label_8 + label_9 + 
                   label_10 + label_11 + label_12 + label_13 + label_14)
        file.write('\n')