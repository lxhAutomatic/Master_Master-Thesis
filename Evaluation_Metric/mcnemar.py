# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:39:21 2023

@author: Xinhao Lan
"""


import numpy as np
from statsmodels.stats.contingency_tables import mcnemar, ExactMcNemar

S_0_0 = # Real negative and Model negative
S_1_0 = # Real positive and Model negative
S_0_1 = # Real negative and Model positive
S_1_1 = # Real positive and Model positive
data = np.array([[S_0_0, S_0_1],[S_1_0, S_1_1]])
result =  ExactMcNemar(data)
print("p-value:", result.pvalue)