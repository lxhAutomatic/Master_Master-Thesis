# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:14:42 2023

@author: Xinhao Lan
"""
def unbalance(R0_P0, R1_P1):
    R0_P1 = 40164 - R0_P0 
    R1_P0 = 4519 - R1_P1

    print('TP:', R1_P1)
    print('FN:', R1_P0)
    print('FP:', R0_P1)
    print('TN:', R0_P0)

    Acc = (R0_P0 + R1_P1)/(R0_P0 + R0_P1 + R1_P0 + R1_P1)
    Rec = R1_P1/(R1_P1 + R1_P0)
    Pre = R1_P1/(R0_P1 + R1_P1)
    FPR = R0_P1/(R0_P0 + R0_P1)
    TPR = R1_P1/(R1_P1 + R1_P0)
    F1 =  (2 * Pre * Rec)/(Pre + Rec)

    print('Accuracy:', Acc)
    print('Recall:', Rec)
    print('Precision:', Pre)
    print('False Positive Rate:', FPR)
    print('True Positive Rate:', TPR)
    print('F1 score:', F1)
    
    pe = (4519*(R0_P1+R1_P1) + 40164*(R0_P0+R1_P0))/(44683*44683)
    p0 = (R0_P0 + R1_P1)/44683
    score = (p0-pe)/(1-pe)
    
    print('score:', score)

def balance(R0_P0, R1_P1):
    R0_P1 = 4519 - R0_P0 
    R1_P0 = 4519 - R1_P1

    print('TP:', R1_P1)
    print('FN:', R1_P0)
    print('FP:', R0_P1)
    print('TN:', R0_P0)

    Acc = (R0_P0 + R1_P1)/(R0_P0 + R0_P1 + R1_P0 + R1_P1)
    Rec = R1_P1/(R1_P1 + R1_P0)
    Pre = R1_P1/(R0_P1 + R1_P1)
    FPR = R0_P1/(R0_P0 + R0_P1)
    TPR = R1_P1/(R1_P1 + R1_P0)
    F1 =  (2 * Pre * Rec)/(Pre + Rec)

    print('Accuracy:', Acc)
    print('Recall:', Rec)
    print('Precision:', Pre)
    print('False Positive Rate:', FPR)
    print('True Positive Rate:', TPR)
    print('F1 score:', F1)
    
    pe = (4519*(R0_P1+R1_P1) + 4519*(R0_P0+R1_P0))/(4519*4519*4)
    p0 = (R0_P0 + R1_P1)/(4519*2)
    score = (p0-pe)/(1-pe)
    
    print('score:', score)
unbalance(29789,1999)
#balance(2549,2476)