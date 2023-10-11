# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 20:22:58 2023

@author: Xinhao Lan
"""

from  PIL import Image
img = Image.open('C:/Users/75581/Desktop/11/DDPM/0049.png')
print(img.size)
box_1 = (0, 0, 256, 256)
box_2 = (256, 0, 512, 256)
box_3 = (512, 0, 768, 256)
box_4 = (768, 0, 1024, 256)
box_5 = (0, 256, 256, 512)
box_6 = (256,256,512,512)
box_7 = (512,256,768,512)
box_8 = (768,256,1024,512)
box_9 = (0, 512, 256, 768)
box_10 = (256, 512, 512, 768)
box_11 = (512,512,768,768)
box_12 = (768,512,1024,768)
box_13 = (0, 768, 256, 1024)
box_14 = (256,768,512,1024)
box_15 = (512,768,768,1024)
box_16 = (768,768,1024,1024)
roi_1 = img.crop(box_1)
roi_2 = img.crop(box_2)
roi_3 = img.crop(box_3)
roi_4 = img.crop(box_4)
roi_5 = img.crop(box_5)
roi_6 = img.crop(box_6)
roi_7 = img.crop(box_7)
roi_8 = img.crop(box_8)
roi_9 = img.crop(box_9)
roi_10 = img.crop(box_10)
roi_11 = img.crop(box_11)
roi_12 = img.crop(box_12)
roi_13 = img.crop(box_13)
roi_14 = img.crop(box_14)
roi_15 = img.crop(box_15)
roi_16 = img.crop(box_16)
for i in range(4):
    up = 256*(i)
    down = 256*(i+1)
    for j in range(4):
        left = 256*j
        right = 256*(j+1)
        box = (left, up, right, down)
        roi = img.crop(box)
        name = i*4 + j + 1
        roi.save('C:/Users/75581/Desktop/11/DDPM/'+str(name)+'.png')
        
