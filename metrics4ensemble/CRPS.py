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
    
    
    N, C, H, W  = X.shape
    
        
            
            
            

    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond)
    X_p[:,0], X_p[:,1] = wc.computeWindDir(X_p[:,0], X_p[:,1])
    angle_dif = wc.angle_diff(X_p[:,1], cond_p[1])


    X_p[:,1] = angle_dif
    cond_p[1,:] = 0.


    crps = ps.crps_ensemble(cond_p,X_p, axis = 0)
   
    return crps
