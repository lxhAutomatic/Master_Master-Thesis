# -*- coding: utf-8 -*-
"""
Created on Wed May 31 23:23:08 2023

@author: Xinhao Lan
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
path = '' # path for the image
chexpert_train_csv = pd.read_csv(path + 'train.csv', sep = ',')
chexpert_valid_csv = pd.read_csv(path + 'valid.csv', sep = ',')
chexpert_train_fit = chexpert_train_csv.fillna(0)
chexpert_valid_fit = chexpert_valid_csv.fillna(0)
chexpert_train_fit = chexpert_train_fit.replace(-1, 1)
chexpert_valid_fit = chexpert_valid_fit.replace(-1, 1)
chexpert_train_df, chexpert_test_df = train_test_split(chexpert_train_fit, test_size = 0.2, random_state = 0, stratify = chexpert_train_fit['Pneumothorax'])
train_1 = chexpert_train_df[chexpert_train_df['Pneumothorax'].isin([1])]
train_0 = chexpert_train_df[chexpert_train_df['Pneumothorax'].isin([0])]

for i in range(train_0.shape[0]):
    temp_rows = (train_0.iloc[i]).copy()
    input_path = 'D:/' + temp_rows['Path']
    img = cv2.imread(input_path)
    img = cv2.resize(img,(389,320))
    output_path_replace = 'CheXpert-v1.0-small-sd/train/train0/' + temp_rows['Path'][26:].replace('/', '_')
    train_0 = train_0.replace(temp_rows['Path'], output_path_replace)
    output_path = 'D:/' + output_path_replace
    cv2.imwrite(output_path, img)
    print('Training dataset labelled 0 img' ,i, 'finished')
train_0.to_csv('D:/CheXpert-v1.0-small-sd/train0.csv',index=None)

for i in range(train_1.shape[0]):
    temp_rows = (train_1.iloc[i]).copy()
    input_path = 'D:/' + temp_rows['Path']
    img = cv2.imread(input_path)
    img = cv2.resize(img,(389,320))
    output_path_replace = 'CheXpert-v1.0-small-sd/train/train1/' + temp_rows['Path'][26:].replace('/', '_')
    train_1 = train_1.replace(temp_rows['Path'], output_path_replace)
    output_path = 'D:/' + output_path_replace
    cv2.imwrite(output_path, img)
    print('Training dataset labelled 1 img' ,i, 'finished')
train_1.to_csv('D:/CheXpert-v1.0-small-sd/train1.csv',index=None)

for i in range(chexpert_test_df.shape[0]):
    temp_rows = (chexpert_test_df.iloc[i]).copy()
    input_path = 'D:/' + temp_rows['Path']
    img = cv2.imread(input_path)
    output_path_replace = 'CheXpert-v1.0-small-sd/test/' + temp_rows['Path'][26:].replace('/', '_')
    chexpert_test_df = chexpert_test_df.replace(temp_rows['Path'], output_path_replace)
    output_path = 'D:/' + output_path_replace
    cv2.imwrite(output_path, img)
    print('Test dataset img' ,i, 'finished')
chexpert_test_df.to_csv('D:/CheXpert-v1.0-small-sd/test.csv',index=None)

for i in range(chexpert_valid_fit.shape[0]):
    temp_rows = (chexpert_valid_fit.iloc[i]).copy()
    input_path = 'D:/' + temp_rows['Path']
    img = cv2.imread(input_path)
    output_path_replace = 'CheXpert-v1.0-small-sd/valid/' + temp_rows['Path'][26:].replace('/', '_')
    chexpert_valid_fit = chexpert_valid_fit.replace(temp_rows['Path'], output_path_replace)
    output_path = 'D:/' + output_path_replace
    cv2.imwrite(output_path, img)
    print('Validation dataset img' ,i, 'finished')
chexpert_valid_fit.to_csv('D:/CheXpert-v1.0-small-sd/valid.csv',index=None)