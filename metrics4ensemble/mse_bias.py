#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:14:51 2023

@author: brochetc
"""
import copy
import numpy as np
import metrics4ensemble.wind_comp as wc
import copy

def mse(cond,X, debiasing = False) :
    
    
    
    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond)

    X_p[:,0], X_p[:,1] = wc.computeWindDir(X_p[:,0], X_p[:,1])
    cond_p[:,0], cond_p[:,1] = wc.computeWindDir(cond_p[:,0], cond_p[:,1])
    
    angle_dif = wc.angle_diff(X_p[:,1], cond_p[:,1])
    #print(angle_dif)
    X_p[:,1] = angle_dif
    cond_p[:,1] = 0.
    
    
    X_mean = X_p.mean(axis=0)
    X_arome_mean = cond_p.mean(axis=0)

    return (X_mean-X_arome_mean)**2.



def bias(cond,X, debiasing = False) :
       
    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond)

        
    X_p[:,0], X_p[:,1] = wc.computeWindDir(X_p[:,0], X_p[:,1])
    cond_p[:,0], cond_p[:,1] = wc.computeWindDir(cond_p[:,0], cond_p[:,1])
    
    angle_dif = wc.angle_diff(X_p[:,1], cond_p[:,1])
    #print(angle_dif)
    X_p[:,1] = angle_dif
    cond_p[:,1] = 0.
    
    X_mean = X_p.mean(axis=0)
    X_arome_mean = cond_p.mean(axis=0)

    return X_mean-X_arome_mean