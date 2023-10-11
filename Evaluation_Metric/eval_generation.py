# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 01:43:58 2023

@author: Xinhao Lan
"""


import torchvision
from pytorch_fid import fid_score
import os
import skimage
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np

real_images_folder = ''
generated_images_folder = ''
def fid_calculate(real_images_folder, generated_images_folder):
    inception_model = torchvision.models.inception_v3(pretrained=True)
    device = 'cuda'
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],batch_size=16,device=device,dims = 2048)
    print('FID value:', fid_value)

def SSIM_calculate(real_images_folder, generated_images_folder):
    path_list_1 = os.listdir(real_images_folder)
    result_ssim = []
    for file in path_list_1:
        img1 = skimage.io.imread(real_images_folder + '/' + file)
        img2 = skimage.io.imread(generated_images_folder + '/' + file)
        result = compare_ssim(img1, img2)
        result_ssim.append(result)
    result_mean = np.mean(result_ssim)
    result_var = np.var(result_ssim)
    return result_mean, result_var

def PSNR_calsulate(real_images_folder, generated_images_folder):
    path_list_1 = os.listdir(real_images_folder)
    result_psnr = []
    for file in path_list_1:
        img1 = skimage.io.imread(real_images_folder + '/' + file)
        img2 = skimage.io.imread(generated_images_folder + '/' + file)
        result = compare_psnr(img1, img2)
        result_psnr.append(result)
    result_mean = np.mean(result_psnr)
    result_var = np.var(result_psnr)
    return result_mean, result_var