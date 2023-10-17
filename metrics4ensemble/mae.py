#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:14:51 2023

@author: brochetc
"""

import numpy as np

def mae(cond,X) :
    
    bias = 0.0
    print(cond.shape)
    N0 = X.shape[0]
    bias = abs(X - cond)
    bias_mean = bias.mean(axis = 0)
    for x in X :
        
        bias = bias + np.abs(x - cond).mean(axis = 0)
    
    return bias_mean
