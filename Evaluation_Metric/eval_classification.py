# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 01:56:44 2023

@author: Xinhao Lan
"""


import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score, confusion_matrix

Path = '' # put the string for the path of the output folder


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
    print('Cohen:', cohen_kappa_score(true_labels.astype(int), temp))
    

calculate (Path)
analysis_broad(Path, 'Pneumothorax Traditional Unbalanced')