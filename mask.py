# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 22:00:03 2023

@author: Xinhao Lan
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

image_path = '' # path for a certain image 
csv_path = '' # path for the csv file
def show_image(flag, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(389,320))
    cv2.imshow("image loaded", img)
    rectangle = np.zeros(img.shape[0:2], dtype = "uint8")
    if flag == 0:
        cv2.rectangle(rectangle, (0,0), (img.shape[1],img.shape[0]), 255, -1)
        pts = np.array([[70, 25], [20, img.shape[0]/2 + 20], [10, img.shape[0] - 30], [img.shape[1]/2 - 15, img.shape[0] - 70], [img.shape[1]/2 - 15, 40]], dtype = np.int32)
        cv2.fillPoly(rectangle, [pts], color = (0,0,0))
        pts = np.array([[img.shape[1] - 70, 25], [img.shape[1] - 20, img.shape[0]/2 + 20], [img.shape[1] - 10, img.shape[0] - 30], [img.shape[1]/2 + 15, img.shape[0] - 70], [img.shape[1]/2 + 15, 40]], dtype = np.int32)
        cv2.fillPoly(rectangle, [pts], color = (0,0,0))
    elif flag == 1:
        cv2.rectangle(rectangle, (0,0), (img.shape[1],img.shape[0]), 255, -1)
        pts = np.array([[80, 20], [309,20], [379,160], [369,300], [20,300], [10,160]], dtype = np.int32)
        cv2.fillPoly(rectangle, [pts], color = (0,0,0))
    elif flag == 2:
        cv2.rectangle(rectangle, (0,0), (img.shape[1],img.shape[0]), 0, -1)
        pts = np.array([[80, 20], [309,20], [379,160], [369,300], [20,300], [10,160]], dtype = np.int32)
        cv2.fillPoly(rectangle, [pts], color = (255,255,255))
    elif flag == 3:
        cv2.rectangle(rectangle, (0,0), (img.shape[1],img.shape[0]), 255, -1)
        cv2.rectangle(rectangle, (20,20), (369,300), 0, -1)
    mask = rectangle
    masked = cv2.bitwise_and(img, img, mask = mask)
    cv2.imshow("Masked image", masked)

# Joint Mask
def mask_1(csv_path):
    chexpert_train_csv = pd.read_csv('', sep = ',')
    chexpert_train_fit = chexpert_train_csv.fillna(0)
    chexpert_train_fit = chexpert_train_fit.replace(-1, 1)
    chexpert_train_df, chexpert_test_df = train_test_split(chexpert_train_fit, test_size = 0.2, random_state = 0, stratify = chexpert_train_fit['Pneumothorax'])
    for i in range(chexpert_train_df.shape[0]):
        temp_rows = (chexpert_train_df.iloc[i]).copy()
        input_path = 'D:/' + temp_rows['Path'] # insert the path for the data folder
        img = cv2.imread(input_path)
        img = cv2.resize(img,(389,320))
        rectangle = np.zeros(img.shape[0:2], dtype = "uint8")
        cv2.rectangle(rectangle, (0,0), (img.shape[1],img.shape[0]), 255, -1)
        pts = np.array([[80, 20], [309,20], [379,160], [369,300], [20,300], [10,160]], dtype = np.int32)
        cv2.fillPoly(rectangle, [pts], color = (0,0,0))
        mask = rectangle
        masked = cv2.bitwise_and(img, img, mask = mask)
        cv2.imwrite(input_path, masked)
        print('img' ,i, 'finished')

# Inverted Joint Mask
def mask_2(csv_path):
    chexpert_train_csv = pd.read_csv('', sep = ',')
    chexpert_train_fit = chexpert_train_csv.fillna(0)
    chexpert_train_fit = chexpert_train_fit.replace(-1, 1)
    chexpert_train_df, chexpert_test_df = train_test_split(chexpert_train_fit, test_size = 0.2, random_state = 0, stratify = chexpert_train_fit['Pleural Effusion'])
    for i in range(chexpert_train_df.shape[0]):
        temp_rows = (chexpert_train_df.iloc[i]).copy()
        input_path = 'D:/' + temp_rows['Path'] # insert the path for the data folder
        img = cv2.imread(input_path)
        img = cv2.resize(img,(389,320))
        rectangle = np.zeros(img.shape[0:2], dtype = "uint8")
        cv2.rectangle(rectangle, (0,0), (img.shape[1],img.shape[0]), 0, -1)
        pts = np.array([[80, 20], [309,20], [379,160], [369,300], [20,300], [10,160]], dtype = np.int32)
        cv2.fillPoly(rectangle, [pts], color = (255,255,255))
        mask = rectangle
        masked = cv2.bitwise_and(img, img, mask = mask)
        cv2.imwrite(input_path, masked)
        print('img' ,i, 'finished')
        
# Square Mask
def mask_3(csv_path):
    chexpert_train_csv = pd.read_csv('', sep = ',')
    chexpert_train_fit = chexpert_train_csv.fillna(0)
    chexpert_train_fit = chexpert_train_fit.replace(-1, 1)
    chexpert_train_df, chexpert_test_df = train_test_split(chexpert_train_fit, test_size = 0.2, random_state = 0, stratify = chexpert_train_fit['Pleural Effusion'])
    for i in range(chexpert_train_df.shape[0]):
        temp_rows = (chexpert_train_df.iloc[i]).copy()
        input_path = 'D:/' + temp_rows['Path'] # insert the path for the data folder
        img = cv2.imread(input_path)
        img = cv2.resize(img,(389,320))
        rectangle = np.zeros(img.shape[0:2], dtype = "uint8")
        cv2.rectangle(rectangle, (0,0), (img.shape[1],img.shape[0]), 255, -1)
        cv2.rectangle(rectangle, (20,20), (369,300), 0, -1)
        mask = rectangle
        masked = cv2.bitwise_and(img, img, mask = mask)
        cv2.imwrite(input_path, masked)
        print('img' ,i, 'finished')

# Separate Mask
def mask_4(csv_path):
    chexpert_train_csv = pd.read_csv('', sep = ',')
    chexpert_train_fit = chexpert_train_csv.fillna(0)
    chexpert_train_fit = chexpert_train_fit.replace(-1, 1)
    chexpert_train_df, chexpert_test_df = train_test_split(chexpert_train_fit, test_size = 0.2, random_state = 0, stratify = chexpert_train_fit['Pneumothorax'])
    for i in range(chexpert_train_df.shape[0]):
        temp_rows = (chexpert_train_df.iloc[i]).copy()
        input_path = '' + temp_rows['Path']  # insert the path for the data folder
        img = cv2.imread(input_path)
        img = cv2.resize(img,(389,320))
        rectangle = np.zeros(img.shape[0:2], dtype = "uint8")
        cv2.rectangle(rectangle, (0,0), (img.shape[1],img.shape[0]), 255, -1)
        pts = np.array([[70, 25], [20, img.shape[0]/2 + 20], [10, img.shape[0] - 30], [img.shape[1]/2 - 15, img.shape[0] - 70], [img.shape[1]/2 - 15, 40]], dtype = np.int32)
        cv2.fillPoly(rectangle, [pts], color = (0,0,0))
        pts = np.array([[img.shape[1] - 70, 25], [img.shape[1] - 20, img.shape[0]/2 + 20], [img.shape[1] - 10, img.shape[0] - 30], [img.shape[1]/2 + 15, img.shape[0] - 70], [img.shape[1]/2 + 15, 40]], dtype = np.int32)
        cv2.fillPoly(rectangle, [pts], color = (0,0,0))
        mask = rectangle
        masked = cv2.bitwise_and(img, img, mask = mask)
        cv2.imwrite(input_path, masked)
        print('img' ,i, 'finished')
