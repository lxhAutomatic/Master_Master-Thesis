# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 00:50:15 2023

@author: Xinhao Lan
"""
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
"""
for i in range(16):
    if i==0:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/1/")
    if i==1:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/2/")
    if i==2:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/3/")
    if i==3:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/4/")
    if i==4:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/5/")
    if i==5:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/6/")
    if i==6:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/7/")
    if i==7:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/8/")
    if i==8:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/9/")
    if i==9:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/10/")
    if i==10:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/11/")
    if i==11:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/12/")
    if i==12:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/13/")
    if i==13:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/14/")
    if i==14:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/15/")
    if i==15:
        img = Image.open("C:/Users/75581/Desktop/11/SD_image/16/")
    axs[i].imshow(img, cmap = 'gray')
    axs[i].set_axis_off()
"""
for i in range(1,17):
    plt.subplot(4,4,i)
    path = 'C:/Users/75581/Desktop/11/SD_image_new/'
    img = Image.open(path + str(i) + '.png').convert("L")
    plt.imshow(img, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
plt.show()