# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 17:44:54 2023

@author: Xinhao Lan
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
import imageio.v2 as imageio
from skimage import io,transform,color
from skimage import img_as_ubyte
from  PIL import Image

def image_matrix(filename):
    x = Image.open(filename)
    data = np.asarray(x)
    return data

def matrix_image(matrix):
    im = Image.fromarray(matrix)
    return im

def PCA(matrix):
    pca = dp.PCA(n_components=3)
    reduced_matrix = pca.fit_transform(matrix)
    print(pca.explained_variance_ratio_)
    return reduced_matrix

def img_PCA(folder_path):
    file_list = os.listdir(folder_path)
    for i,name in enumerate(file_list):
        input_path = folder_path + name
        img = imageio.imread(input_path)
        img_new = color.rgb2gray(img).flatten()
        if i==0:
            num = img_new
        else:
            num = np.vstack((num, img_new))
    matrix = PCA(num)
    return matrix


#matrix_1 = img_PCA('D:/CheXpert-v1.0-small-sd/train/train1/')
#matrix_0 = img_PCA('D:/CheXpert-v1.0-small-sd/train/train0/')
matrix_1 = img_PCA('') # folder for images with label 1
matrix_0 = img_PCA('') # folder for images with label 0
for i in range(matrix_1.shape[0]):
    plt.scatter(matrix_1[i][0], matrix_1[i][1], color = 'red')
for i in range(matrix_0.shape[0]):
    plt.scatter(matrix_0[i][0], matrix_0[i][1], color = 'green')