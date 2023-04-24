#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:38:11 2023

@author: brochetc

AROME-specific version of CRPS

"""

import properscoring as ps
import numpy as np



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
    
    avg_crps = np.zeros((C, H, W))
    
    for x in X :
        
        avg_crps = avg_crps + ps.crps_ensemble(x, cond, axis = 0) # giving the 'ensemble axis'
    
    return avg_crps / N
