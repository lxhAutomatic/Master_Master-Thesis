# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:04:05 2023

@author: Xinhao Lan
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CreateDatasetFromImages(Dataset):
    def __init__(self, csv_path, file_path, resize_height=256, resize_width=256):
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.file_path = file_path
        self.to_tensor = transforms.ToTensor()

        self.data_info = pd.read_csv(csv_path, sep = ',')
        self.image_arr = np.asarray(self.data_info.iloc[1:, 0])  
        self.label_arr = np.asarray(self.data_info.iloc[1:, 14])
        
        self.data_len = len(self.data_info.index) - 1

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(self.file_path + single_image_name)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])
        img_as_img = transform(img_as_img)
        label = self.label_arr[index]
        return (img_as_img, label)
        
    def __len__(self):
        return self.data_len

csv_path = 'D:/CheXpert-v1.0-small/train.csv'
file_path = 'D:/'

MyTrainDataset = CreateDatasetFromImages(csv_path , file_path)
train_loader = torch.utils.data.DataLoader(
        dataset=MyTrainDataset,
        batch_size=1, 
        shuffle=False,
    )
for i, data in enumerate(train_loader):
    if i == 0:
        print("第 {} 个Batch \n{}".format(i, data))
        print(data[0])
    else:
        break
