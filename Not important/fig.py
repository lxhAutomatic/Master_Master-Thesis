# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 00:44:16 2023

@author: Xinhao Lan
"""
import matplotlib.pyplot as plt



epoch = [1,2,3,4,5,6,7,8,9,10]
train_acc =  [0.529,0.684,0.718,0.735,0.749,0.760,0.770,0.779,0.788,0.798]
val_acc =    [0.653,0.702,0.714,0.727,0.732,0.736,0.738,0.740,0.742,0.740]
train_loss = [0.327,0.179,0.149,0.134,0.124,0.116,0.109,0.104,0.098,0.093]
val_loss =   [0.208,0.163,0.149,0.139,0.135,0.132,0.130,0.130,0.129,0.129]
plt.figure(figsize=(12,8), dpi=100)
plt.plot(epoch, train_loss, c='red', label = 'Train')
plt.plot(epoch, val_loss, c='blue', label = 'Val')
plt.scatter(epoch, train_loss, c='red')
plt.scatter(epoch, val_loss, c='blue')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("Epochs", fontdict={'size': 16})
plt.ylabel("Loss", fontdict={'size': 16})
plt.title("PCS + Plus + Bert + Unfreeze 1 layer", fontdict={'size': 20})
plt.show()

plt.figure(figsize=(12,8), dpi=100)
plt.plot(epoch, train_acc, c='red', label = 'Train')
plt.plot(epoch, val_acc, c='blue', label = 'Val')
plt.scatter(epoch, train_acc, c='red')
plt.scatter(epoch, val_acc, c='blue')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("Epochs", fontdict={'size': 16})
plt.ylabel("Accuracy", fontdict={'size': 16})
plt.title("PCS + Plus + Bert + Unfreeze 1 layer", fontdict={'size': 20})
plt.show()