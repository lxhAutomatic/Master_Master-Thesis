# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:03:38 2023

@author: Xinhao Lan
"""

from keras_preprocessing import image
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import cv2
#import matplotlib.pyplot as plt

def flipping(num):
    P = [0, 1, 2, 3, 4, 5, 6, 7]
    flag = random.choice(P)
    if flag == 0 or flag == 1 or flag == 2:
        num = np.flipud(num)
    elif flag == 3 or flag == 4 or flag == 5:
        num = np.fliplr(num)
    return num

def rotation(num, degree, fill):
    num = image.random_rotation(num, degree, row_axis = 0, col_axis = 1,channel_axis = 2, fill_mode = fill)
    return num

def shifting(num, wrg, hrg, fill):
    num = image.random_shift(num, wrg, hrg, row_axis = 0, col_axis = 1,channel_axis = 2, fill_mode = fill)
    return num

def shearing(num, intensity, fill):
    num = image.random_shear(num, intensity, row_axis = 0, col_axis = 1,channel_axis = 2, fill_mode = fill)
    return num
    
def zooming(num, zoom_range, fill):
    num = image.random_zoom(num, zoom_range, row_axis = 0, col_axis = 1,channel_axis = 2, fill_mode = fill)
    return num

def cropping(img, flag):
    #P = [0, 1, 2, 3]
    #flag = random.choice(P)
    if flag == 0:
        img = transforms.CenterCrop(300)(img)
    elif flag == 1:
        img = transforms.RandomCrop(300)(img)
    elif flag == 2:
        img = transforms.RandomResizedCrop(300)(img)
    return img

def coloring(img):
    P = [0, 1]
    flag = random.choice(P)
    brightness = (1,1)
    contrast = (1,1)
    saturation = (1,1)
    if flag == 0:
        brightness = (1,2)
    elif flag == 1:
        contrast = (1,2)
    elif flag == 2:
        saturation = (1,2)
    img = transforms.ColorJitter(brightness, contrast, saturation)(img)
    return img

def blurring(img, sigma, flag):
    P = [0, 1, 2]
    flag = random.choice(P)
    flag = 1
    if flag == 0:
        kernel_size = (5, 5)
        img = transforms.GaussianBlur(kernel_size, sigma)(img)
    elif flag == 1:
        kernel_size = (13, 13)
        img = transforms.GaussianBlur(kernel_size, sigma)(img)
    elif flag == 2:
        kernel_size = (25, 25)
        img = transforms.GaussianBlur(kernel_size, sigma)(img)
    return img

class AddPepperNoise(object):
    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w), p=[signal_pct, noise_pct/2., noise_pct/2.])
            img_[mask == 1] = 255 
            img_[mask == 2] = 0  
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img = N + img
            img[img > 255] = 255                     
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img
        
def noise(img, mean, variance, amplitude, p1, p2, flag):
    #P = [0, 1]
    #flag = random.choice(P)
    if flag == 0:
        img_new = AddGaussianNoise(mean, variance, amplitude)(img)
    elif flag == 1:
        img_new = AddPepperNoise(p1, p2)(img)
    return img_new

def create_1(flag, path):
    chexpert_train_csv = pd.read_csv(path + 'train.csv', sep = ',')
    chexpert_valid_csv = pd.read_csv(path + 'valid.csv', sep = ',')
    chexpert_train_fit = chexpert_train_csv.fillna(0)
    chexpert_valid_fit = chexpert_valid_csv.fillna(0)
    chexpert_train_fit = chexpert_train_fit.replace(-1, 1)
    chexpert_valid_fit = chexpert_valid_fit.replace(-1, 1)
    chexpert_train_df_2, chexpert_test_df_2 = train_test_split(chexpert_train_fit, test_size = 0.1, random_state = 0, stratify = chexpert_train_fit['Pneumothorax'])
    train_1 = chexpert_train_df_2[chexpert_train_df_2['Pneumothorax'].isin([1])]
    train_0 = chexpert_train_df_2[chexpert_train_df_2['Pneumothorax'].isin([0])]
    train_data_1 = chexpert_train_fit[chexpert_train_fit['Pneumothorax'].isin([1])]
    train_data_0 = chexpert_train_fit[chexpert_train_fit['Pneumothorax'].isin([0])]
    
    df1 = train_data_1.sample(frac = 0.5)
    df2 = train_data_1[~train_data_1.index.isin(df1.index)]
    df0 = train_data_0.sample(frac = 0.05)
    df3 = train_data_0[~train_data_0.index.isin(df0.index)]
    chexpert_test_df_1 = pd.concat([df0, df1])
    chexpert_train_df_1 = pd.concat([df2, df3])
    for i in range(train_1.shape[0]):
        temp = 1
        temp_rows = (train_1.iloc[i]).copy()
        print('img', i)
        while(temp<12):
            input_path = path + temp_rows['Path'][19:]
            img = image.load_img(input_path)
            num = image.img_to_array(img)
            if flag == 1:
                num = flipping(num)
            elif flag == 2:
                num = rotation(num, 90, 'nearest')
            elif flag == 3:
                num = shifting(num, 0.2, 0.2, 'constant')
            elif flag == 4:
                num = shearing(num, 30, 'nearest')
            elif flag == 5:
                img = cropping(img, 0)
            elif flag == 6:
                img = coloring(img)
            elif flag == 7:
                img = blurring(img, 10, 1)
            elif flag == 8:
                img = noise(img, random.uniform(0.5,1.5), 0.5, random.uniform(0, 30), 0.99, 1.0, 0)
            output_path_1 = 'CheXpert-v1.0-small/train/1_' + str(i) + '_' + str(temp) +'.jpg'
            output_path_2 = path + 'train/1_' + str(i) + '_' + str(temp) +'.jpg'
            temp_rows['Path'] = output_path_1
            chexpert_train_csv.loc[223413+i*11+temp] = temp_rows
            chexpert_train_df_2 = chexpert_train_df_2.append(temp_rows)
            chexpert_train_df_1 = chexpert_train_df_1.append(temp_rows)
            image.save_img(output_path_2, num)
            temp = temp + 1
            print('    Augmented img', temp)
    mid = 223413+i*11+temp-1
    for i in range(int(train_0.shape[0]/3)):
        temp = 1
        temp_rows = (train_0.iloc[i]).copy()
        print('img', i)
        while(temp<2):
            input_path = path + temp_rows['Path'][19:]
            img = image.load_img(input_path)
            num = image.img_to_array(img)
            if flag == 1:
                num = flipping(num)
            elif flag == 2:
                num = rotation(num, 90, 'nearest')
            elif flag == 3:
                num = shifting(num, 0.2, 0.2, 'constant')
            elif flag == 4:
                num = shearing(num, 30, 'nearest')
            elif flag == 5:
                img = cropping(img)
            elif flag == 6:
                img = coloring(img)
            elif flag == 7:
                img = blurring(img, 10)
            elif flag == 8:
                img = noise(img, random.uniform(0.5,1.5), 0.5, random.uniform(0, 30), 0.99, 1.0)
            output_path_1 = 'CheXpert-v1.0-small/train/0_' + str(i) + '_' + str(temp) +'.jpg'
            output_path_2 = path + 'train/0_' + str(i) + '_' + str(temp) +'.jpg'
            temp_rows['Path'] = output_path_1
            chexpert_train_csv.loc[mid+i*1+temp] = temp_rows
            chexpert_train_df_2 = chexpert_train_df_2.append(temp_rows)
            chexpert_train_df_1 = chexpert_train_df_1.append(temp_rows)
            image.save_img(output_path_2, num)
            temp = temp + 1
            print('    Augmented img', temp)
    chexpert_train_csv.to_csv(path + 'final_train_test.csv', index=None)
    chexpert_train_df_2.to_csv(path + 'final_train_unbalance.csv', index=None)
    chexpert_train_df_1.to_csv(path + 'final_train_balance.csv', index=None)
    chexpert_test_df_1.to_csv(path + 'final_test_balance.csv', index=None)
    chexpert_test_df_2.to_csv(path + 'final_test_unbalance.csv', index=None)

def create_2(flag, path):
    chexpert_train_csv = pd.read_csv(path + 'train.csv', sep = ',')
    chexpert_valid_csv = pd.read_csv(path + 'valid.csv', sep = ',')
    chexpert_train_fit = chexpert_train_csv.fillna(0)
    chexpert_valid_fit = chexpert_valid_csv.fillna(0)
    chexpert_train_fit = chexpert_train_fit.replace(-1, 1)
    chexpert_valid_fit = chexpert_valid_fit.replace(-1, 1)
    chexpert_train_df_2, chexpert_test_df_2 = train_test_split(chexpert_train_fit, test_size = 0.1, random_state = 0, stratify = chexpert_train_fit['Pneumothorax'])
    train_1 = chexpert_train_df_2[chexpert_train_df_2['Pneumothorax'].isin([1])]
    train_0 = chexpert_train_df_2[chexpert_train_df_2['Pneumothorax'].isin([0])]
    train_data_1 = chexpert_train_fit[chexpert_train_fit['Pneumothorax'].isin([1])]
    train_data_0 = chexpert_train_fit[chexpert_train_fit['Pneumothorax'].isin([0])]
    
    train_0 = train_data_0.smaple(n=train_data_1.shape[0])
    train_1 = train_data_1
    chexpert_test_df_1 = train_0
    chexpert_train_df_1 = train_1
    for i in range(train_1.shape[0]):
        temp = 1
        temp_rows = (train_1.iloc[i]).copy()
        print('img', i)
        while(temp<12):
            input_path = path + temp_rows['Path'][19:]
            img = image.load_img(input_path)
            num = image.img_to_array(img)
            if flag == 1:
                num = flipping(num)
            elif flag == 2:
                num = rotation(num, 90, 'nearest')
            elif flag == 3:
                num = shifting(num, 0.2, 0.2, 'constant')
            elif flag == 4:
                num = shearing(num, 30, 'nearest')
            elif flag == 5:
                img = cropping(img, 0)
            elif flag == 6:
                img = coloring(img)
            elif flag == 7:
                img = blurring(img, 10, 1)
            elif flag == 8:
                img = noise(img, random.uniform(0.5,1.5), 0.5, random.uniform(0, 30), 0.99, 1.0, 0)
            output_path_1 = 'CheXpert-v1.0-small/train/1_' + str(i) + '_' + str(temp) +'.jpg'
            output_path_2 = path + 'train/1_' + str(i) + '_' + str(temp) +'.jpg'
            temp_rows['Path'] = output_path_1
            chexpert_train_csv.loc[2*train_1.shape[0]+i*11+temp] = temp_rows
            chexpert_train_df_2 = chexpert_train_df_2.append(temp_rows)
            chexpert_train_df_1 = chexpert_train_df_1.append(temp_rows)
            image.save_img(output_path_2, num)
            temp = temp + 1
            print('    Augmented img', temp)
    mid = 2*train_1.shape[0]+i*11+temp-1
    for i in range(int(train_0.shape[0]/3)):
        temp = 1
        temp_rows = (train_0.iloc[i]).copy()
        print('img', i)
        while(temp<2):
            input_path = path + temp_rows['Path'][19:]
            img = image.load_img(input_path)
            num = image.img_to_array(img)
            if flag == 1:
                num = flipping(num)
            elif flag == 2:
                num = rotation(num, 90, 'nearest')
            elif flag == 3:
                num = shifting(num, 0.2, 0.2, 'constant')
            elif flag == 4:
                num = shearing(num, 30, 'nearest')
            elif flag == 5:
                img = cropping(img)
            elif flag == 6:
                img = coloring(img)
            elif flag == 7:
                img = blurring(img, 10)
            elif flag == 8:
                img = noise(img, random.uniform(0.5,1.5), 0.5, random.uniform(0, 30), 0.99, 1.0)
            output_path_1 = 'CheXpert-v1.0-small/train/0_' + str(i) + '_' + str(temp) +'.jpg'
            output_path_2 = path + 'train/0_' + str(i) + '_' + str(temp) +'.jpg'
            temp_rows['Path'] = output_path_1
            chexpert_train_csv.loc[mid+i*1+temp] = temp_rows
            chexpert_train_df_2 = chexpert_train_df_2.append(temp_rows)
            chexpert_train_df_1 = chexpert_train_df_1.append(temp_rows)
            image.save_img(output_path_2, num)
            temp = temp + 1
            print('    Augmented img', temp)
    chexpert_train_csv.to_csv(path + 'final_train_test.csv', index=None)
    chexpert_train_df_2.to_csv(path + 'final_train_unbalance.csv', index=None)
    chexpert_train_df_1.to_csv(path + 'final_train_balance.csv', index=None)
    chexpert_test_df_1.to_csv(path + 'final_test_balance.csv', index=None)
    chexpert_test_df_2.to_csv(path + 'final_test_unbalance.csv', index=None)