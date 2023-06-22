"""


AROME-specific version of skill_spread

"""

import numpy as np
import wind_comp as wc
import copy


def bias_ens(cond, X):
    """
    
    Inputs :
        
        X : N x C x H x W array with N samples
        
        cond : C x H x W array observation
        
    Returns :
        
        bias : avg(X) - cond  
    
    """
    
    N, C, H, W  = X.shape
    
    
    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond)    
    X_p[:,0], X_p[:,1] = wc.computeWindDir(X_p[:,0], X_p[:,1])
    angle_dif = wc.angle_diff(X_p[:,1], cond_p[1])
    print(angle_dif)
    X_p[:,1] = angle_dif
    cond_p[1,~np.isnan(cond_p[1])] = 0.
    
    
    X_p_mean = X_p.mean(axis=0)
    X_bias = X_p_mean - cond_p

    return X_bias
