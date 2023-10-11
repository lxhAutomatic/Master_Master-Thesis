# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:17:21 2023

@author: Xinhao Lan
"""
import numpy as np
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
"""
img1 = cv2.imread('C:/Users/75581/Desktop/11/original_image/1.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
(h,w)=img1.shape[:2]
print(h,w)
img2 = cv2.imread('C:/Users/75581/Desktop/11/DDIM/1.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
resized=cv2.resize(img2,(w,h))
(h1,w1)=resized.shape[:2]
print(h1,w1)

mssim = compare_ssim(img1, resized)
print(mssim)
mpsnr = compare_psnr(img1, resized)
print(mpsnr)
"""
arr=[]
for i in range (16):
    img1 = cv2.imread('') # path for image 1
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    (h,w)=img1.shape[:2]
    img2 = cv2.imread('') # path for image 2
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(img2,(w,h))
    mpsnr = compare_psnr(img1, resized)
    print(mpsnr)
    if mpsnr != float('inf'):
        arr.append(mpsnr)

arr_mean = np.mean(arr)
arr_var = np.var(arr)
arr_std = np.std(arr,ddof=1)
print("Mean value：%f" % arr_mean)
print("Variance：%f" % arr_var)
print("Standard Deviation:%f" % arr_std)
