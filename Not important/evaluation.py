# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:01:31 2023

@author: Xinhao Lan
"""

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
Path_1 = 'C:/Users/75581/Desktop/Result/Pleural Effusion_baseline/'
Path_2 = 'C:/Users/75581/Desktop/Result/Pleural Effusion_inverted-joint-mask/'
Path_3 = 'C:/Users/75581/Desktop/Result/Pleural Effusion_joint-mask/'
Path_4 = 'C:/Users/75581/Desktop/Result/Pleural Effusion_square-mask/'

Path_5 = 'C:/Users/75581/Desktop/Result/Pneumothorax_baseline/'
Path_6 = 'C:/Users/75581/Desktop/Result/Pneumothorax_inverted-joint-mask/'
Path_7 = 'C:/Users/75581/Desktop/Result/Pneumothorax_joint-mask/'
Path_8 = 'C:/Users/75581/Desktop/Result/Pneumothorax_square-mask/'

Path_9 = 'C:/Users/75581/Desktop/Result/augmentation-1/Blurring/'
Path_10 = 'C:/Users/75581/Desktop/Result/augmentation-1/Coloring/'
Path_11 = 'C:/Users/75581/Desktop/Result/augmentation-1/Cropping/'
Path_12 = 'C:/Users/75581/Desktop/Result/augmentation-1/Flipping/'
Path_13 = 'C:/Users/75581/Desktop/Result/augmentation-1/Noise/'
Path_14 = 'C:/Users/75581/Desktop/Result/augmentation-1/Rotation/'
Path_15 = 'C:/Users/75581/Desktop/Result/augmentation-1/Shearing/'
Path_16 = 'C:/Users/75581/Desktop/Result/augmentation-1/Shifting/'

Path_17 = 'C:/Users/75581/Desktop/Result/augmentation-2/Blurring/'
Path_18 = 'C:/Users/75581/Desktop/Result/augmentation-2/Coloring/'
Path_19 = 'C:/Users/75581/Desktop/Result/augmentation-2/Cropping/'
Path_20 = 'C:/Users/75581/Desktop/Result/augmentation-2/Flipping/'
Path_21 = 'C:/Users/75581/Desktop/Result/augmentation-2/Noise/'
Path_22 = 'C:/Users/75581/Desktop/Result/augmentation-2/Rotation/'
Path_23 = 'C:/Users/75581/Desktop/Result/augmentation-2/Shearing/'
Path_24 = 'C:/Users/75581/Desktop/Result/augmentation-2/Shifting/'

Path_25 = 'C:/Users/75581/Desktop/Result/pl/'
Path_26 = 'C:/Users/75581/Desktop/Result/pn/'
Path_27 = 'C:/Users/75581/Desktop/Result/baseline_result_4/'

def calculate (path):
    thresholds_list = np.loadtxt(path + 'validation_thresholds_list.txt')
    pred_labels = np.loadtxt(path + 'pred_labels.txt')
    true_labels = np.loadtxt(path + 'true_labels.txt')
    acc_list = []
    pre_list = []
    f1_list = []
    rec_list = []
    temp = pred_labels.copy()
    for i in range(len(thresholds_list)):
        temp[pred_labels>thresholds_list[i]]=1
        temp[pred_labels<=thresholds_list[i]]=0
        acc_list.append(accuracy_score(true_labels.astype(int), temp))
        pre_list.append(precision_score(true_labels.astype(int), temp, zero_division=0))
        rec_list.append(recall_score(true_labels.astype(int), temp, zero_division=0))
        f1_list.append(f1_score(true_labels.astype(int), temp, zero_division=0))
    
    np.savetxt(path + 'acc_list.txt', acc_list, fmt = '%f')
    np.savetxt(path + 'pre_list.txt', pre_list, fmt = '%f')
    np.savetxt(path + 'rec_list.txt', rec_list, fmt = '%f')
    np.savetxt(path + 'f1_list.txt', f1_list, fmt = '%f')
    
    print(thresholds_list[acc_list.index(max(acc_list))])
    print(max(acc_list))
    print(thresholds_list[pre_list.index(max(pre_list))])
    print(max(pre_list))
    print(thresholds_list[rec_list.index(max(rec_list))])
    print(max(rec_list))
    print(thresholds_list[f1_list.index(max(f1_list))])
    print(max(f1_list))
    threshold = thresholds_list[f1_list.index(max(f1_list))]
    temp = pred_labels.copy()
    temp[pred_labels>threshold]=1
    temp[pred_labels<=threshold]=0
    print(confusion_matrix(true_labels, temp, labels = [0,1]))
    
def analysis_broad (path, str1):
    x = np.loadtxt(path + 'validation_thresholds_list.txt')
    y_acc = (np.loadtxt(path + 'acc_list.txt')).tolist()
    y_pre = (np.loadtxt(path + 'pre_list.txt')).tolist()
    y_rec = (np.loadtxt(path + 'rec_list.txt')).tolist()
    y_f1 = (np.loadtxt(path + 'f1_list.txt')).tolist()
    thresholds_list = np.loadtxt(path + 'thresholds_list.txt')
    pred_labels = np.loadtxt(path + 'pred_labels.txt')
    true_labels = np.loadtxt(path + 'true_labels.txt')
    print(sum(true_labels))
    print(len(true_labels))
    print('Accuracy threshold:', thresholds_list[y_acc.index(max(y_acc))])
    print('Accuracy:', max(y_acc))
    print('Precision threshold:', thresholds_list[y_pre.index(max(y_pre))])
    print('Precision:', max(y_pre))
    print('Recall threshold:', thresholds_list[y_rec.index(max(y_rec))])
    print('Recall:', max(y_rec))
    print('F1 threshold:', thresholds_list[y_f1.index(max(y_f1))])
    print('F1:', max(y_f1))
    threshold = thresholds_list[y_f1.index(max(y_f1))]
    temp = pred_labels.copy()
    temp[pred_labels>threshold]=1
    temp[pred_labels<=threshold]=0
    cm = confusion_matrix(true_labels, temp, labels = [0,1])
    tn, fp, fn, tp = cm.ravel()
    print('Confusion matrix:\n', cm)
    print('TP:', tp)
    print('TN:', tn)
    print('FP:', fp)
    print('FN:', fn)
    print('TPR:', tp/(tp+fn))
    print('FPR:', fp/(tn+fp))
    print('Accuracy:', accuracy_score(true_labels.astype(int), temp))
    print('Precision:', precision_score(true_labels.astype(int), temp, zero_division=0))
    print('Recall:', recall_score(true_labels.astype(int), temp, zero_division=0))
    print('F1:', f1_score(true_labels.astype(int), temp, zero_division=0))
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(str1)
    axs[0][0].set_title("Accuracy")
    axs[0][0].set_xlim((0.0, 1.0))
    axs[0][0].set_xlabel('Threshold value')
    axs[0][0].set_ylabel('Result')
    axs[0][0].plot(x, y_acc, ls="-.",color="r",marker =",", lw=1)
    
    axs[0][1].set_title("Precision")
    axs[0][1].set_xlim((0.0, 1.0))
    axs[0][1].set_xlabel('Threshold value')
    axs[0][1].set_ylabel('Result')
    axs[0][1].plot(x, y_pre, ls="-.",color="r",marker =",", lw=1)
    
    axs[1][0].set_title("Recall")
    axs[1][0].set_xlim((0.0, 1.0))
    axs[1][0].set_xlabel('Threshold value')
    axs[1][0].set_ylabel('Result')
    axs[1][0].plot(x, y_rec, ls="-.",color="r",marker =",", lw=1)
    
    axs[1][1].set_title("F1 Score")
    axs[1][1].set_xlim((0.0, 1.0))
    axs[1][1].set_xlabel('Threshold value')
    axs[1][1].set_ylabel('Result')
    axs[1][1].plot(x, y_f1, ls="-.",color="r",marker =",", lw=1)
    """

def heatmap(path):
    image_path = 'D:/CheXpert-v1.0-small/valid/patient64561/study1/view1_frontal.jpg'
    def preprocess(x):
        x = tf.io.read_file(x)
        x = tf.image.decode_jpeg(x, channels=3)
        x = tf.image.resize(x, [320,389])
        x =tf.expand_dims(x, 0) 
        return x
    input_shape = (320, 389, 3)

    img = preprocess(image_path)
    all_pathologies = ['Pleural Effusion']
    img_input = Input(shape = input_shape)
    base_model = DenseNet121(include_top=False, input_tensor=img_input, input_shape=input_shape, pooling="max", weights='imagenet')
    x = base_model.output
    predictions = Dense(len(all_pathologies), activation="sigmoid", name="predictions")(x)
    model = Model(inputs=img_input, outputs=predictions)
    #model.load_weights(path + 'best_weight.h5')
    # model = Model(inputs=img_input, outputs=x)
    Predictions = model.predict(img)
    print(Predictions)

    from tensorflow.keras import models
    last_conv_layer = model.get_layer('conv5_block16_2_conv')
    heatmap_model = models.Model([model.inputs], [last_conv_layer.output, model.output])

    import tensorflow.keras.backend as K
    with tf.GradientTape() as gtape:
        conv_output, Predictions = heatmap_model(img)
        prob = Predictions[:, np.argmax(Predictions[0])]
        grads = gtape.gradient(prob, conv_output)
        pooled_grads = K.mean(grads, axis=(0,1,2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    plt.matshow(heatmap[0], cmap='viridis')
    import cv2
    original_img=cv2.imread(image_path)
    heatmap1 = cv2.resize(heatmap[0], (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap1 = np.uint8(255*heatmap1)
    heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
    frame_out=cv2.addWeighted(original_img,0.5,heatmap1,0.5,0)
    cv2.imshow('heatmap', frame_out)
    #cv2.imwrite('Egyptian_cat.jpg', frame_out)

    #plt.figure()
    #plt.imshow(frame_out)
    
# calculate (Path_1)
# calculate (Path_2)
# calculate (Path_3)
# calculate (Path_4)

# calculate (Path_5)
# calculate (Path_6)
# calculate (Path_7)
# calculate (Path_8)

 


    
# analysis_broad(Path_1, 'Pleural Effusion Baseline')
# analysis_broad(Path_2, 'Pleural Effusion Inverted joint mask')
# analysis_broad(Path_3, 'Pleural Effusion Joint mask')
# analysis_broad(Path_4, 'Pleural Effusion Square mask')

# analysis_broad(Path_5, 'Pneumothorax Baseline')
# analysis_broad(Path_6, 'Pneumothorax Inverted joint mask')
# analysis_broad(Path_7, 'Pneumothorax Joint mask')
# analysis_broad(Path_8, 'Pneumothorax Square mask')

#calculate (Path_9)
#analysis_broad(Path_9, 'Pneumothorax Traditional Balanced')
#calculate (Path_17)
#analysis_broad(Path_17, 'Pneumothorax Traditional Balanced')

#calculate (Path_10)
#analysis_broad(Path_10, 'Pneumothorax Traditional Balanced')
#calculate (Path_18)
#analysis_broad(Path_18, 'Pneumothorax Traditional Balanced')

#calculate (Path_11)
#analysis_broad(Path_11, 'Pneumothorax Traditional Balanced')
#calculate (Path_19)
#analysis_broad(Path_19, 'Pneumothorax Traditional Balanced')

#calculate (Path_12)
#analysis_broad(Path_12, 'Pneumothorax Traditional Balanced')
#calculate (Path_20)
#analysis_broad(Path_20, 'Pneumothorax Traditional Balanced')

# calculate (Path_13)
# analysis_broad(Path_13, 'Pneumothorax Traditional Unbalanced')
# calculate (Path_21)
# analysis_broad(Path_21, 'Pneumothorax Traditional Unbalanced')

#calculate (Path_14)
#analysis_broad(Path_14, 'Pneumothorax Traditional Unbalanced')
#alculate (Path_22)
#analysis_broad(Path_22, 'Pneumothorax Traditional Unbalanced')

#calculate (Path_15)
#analysis_broad(Path_15, 'Pneumothorax Traditional Unbalanced')
#calculate (Path_23)
#analysis_broad(Path_23, 'Pneumothorax Traditional Unbalanced')

# calculate (Path_16)  
# analysis_broad(Path_16, 'Pneumothorax Traditional Unbalanced')
# calculate (Path_24)  
# analysis_broad(Path_24, 'Pneumothorax Traditional Unbalanced')

calculate (Path_25)
analysis_broad(Path_25, 'Pneumothorax Traditional Unbalanced')
calculate (Path_26)
analysis_broad(Path_26, 'Pneumothorax Traditional Unbalanced')


"""
true_labels = [0,0,0,0,0,0,0,0,0,0,0]
pred_labels = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
thresholds = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
temp = pred_labels.copy()
for i in range(len(thresholds)):
    temp_value = thresholds[i]
    print(temp_value)
    for j in range(len(pred_labels)):
        if pred_labels[j]>temp_value:
            temp[j] = 1
        else:
            temp[j] = 0
    print(pred_labels)
    print(temp)
"""