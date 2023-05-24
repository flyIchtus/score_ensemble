#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:38:11 2023

@author: brochetc

AROME-specific version of CRPS

"""

import properscoring as ps
import numpy as np
import wind_comp as wc
import copy
def ensemble_crps(cond, X):
    """
    'average CRPS' for each member of the 'X' ensemble, compared to the distribution induced by condition
    
    Inputs :
        
        X : N x C x H x W array with N samples
        
        cond : N_c x C x H x W array with N_c members
        
    Returns :
        
        avg_crps : C x H x W array containing the result
    
    """
    
    
    #cond = np.expand_dims(cond, axis = 0)
    #print(np.nanmax(cond[2]), X[:,2].max())
    N, C, H, W  = X.shape
    
    avg_crps = np.zeros((C, H, W))
        
            
            
            


    X[:,0], X[:,1] = wc.computeWindDir(X[:,0], X[:,1])
    angle_dif = wc.angle_diff(X[:,1], cond[1])


    X[:,1] = angle_dif
    cond[1,:] = 0.
    #angle_dif = wc.angle_diff(X[:,1], cond[1])
    #var1 = np.array([3.])
    #var2 = np.array([360.])
    #print(wc.angle_diff(var1, var2))
 
    #for i in range(H):
        
        #for j in range(W):
            
            #print(cond[1,i,j], X[0,1,i,j], angle_dif[0,i,j],)   


    crps = ps.crps_ensemble(cond,X, axis = 0)
    
    print("Mean CRPS Ens", np.nanmean(crps[2]), np.nanmean(crps[1]), np.nanmean(crps[0]))
    
    #print(X[1,2])
    # for x in X :
        
    #     avg_crps = avg_crps + ps.crps_ensemble(x, cond, axis = 0) # giving the 'ensemble axis'
        
    return
