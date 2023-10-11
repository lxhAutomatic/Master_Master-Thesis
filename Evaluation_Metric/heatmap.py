# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:27:42 2023

@author: Xinhao Lan
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import models
import tensorflow.keras.backend as K

image_path = 'xxxxxxxxxxx' #use the file path
input_shape = (389, 320, 3)
def preprocess(x):
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3)
    x = tf.image.resize(x, [389,320])
    x =tf.expand_dims(x, 0) 
    return x

def heatmap(image_path, input_shape):
    img = preprocess(image_path)
    all_pathologies = ['Pneumothorax']
    img_input = Input(shape = input_shape)
    base_model = DenseNet121(include_top=False, input_tensor=img_input, input_shape=input_shape, pooling="max", weights='imagenet')
    x = base_model.output
    predictions = Dense(len(all_pathologies), activation="sigmoid", name="predictions")(x)
    model = Model(inputs=img_input, outputs=predictions)
    model.load_weights('')
    
    Predictions = model.predict(img)
    print(Predictions)
    
    
    last_conv_layer = model.get_layer('conv5_block16_2_conv')
    heatmap_model = models.Model([model.inputs], [last_conv_layer.output, model.output])
    
    with tf.GradientTape() as gtape:
        conv_output, Predictions = heatmap_model(img)
        prob = Predictions[:, np.argmax(Predictions[0])]
        grads = tf.Gradient(prob, conv_output)
        pooled_grads = K.mean(grads, axis=(0,1,2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    plt.matshow(heatmap[0], cmap='viridis')
    
    original_img=cv2.imread(image_path)
    heatmap1 = cv2.resize(heatmap[0], (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap1 = np.uint8(255*heatmap1)
    heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
    frame_out=cv2.addWeighted(original_img,0.5,heatmap1,0.5,0)
    
    plt.figure()
    plt.imshow(frame_out)



