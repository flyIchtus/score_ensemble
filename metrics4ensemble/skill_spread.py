"""


AROME-specific version of skill_spread

"""

import numpy as np
import wind_comp as wc
import copy

def skill_spread(cond, X):
    """
    
    Inputs :
        
        X : N x C x H x W array with N samples
        
        cond : C x H x W array observation
        
    Returns :
        
        skill spread  :2 x C x H x W array containing the result 0 is skill and 1 is spread
    
    """
    N, C, H, W  = X.shape
    
    sp_out = np.zeros((2,C,H,W))
    

    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond)   

    
    X_p[:,0], X_p[:,1] = wc.computeWindDir(X_p[:,0], X_p[:,1])
    angle_dif = wc.angle_diff(X_p[:,1], cond_p[1])
    X_p[:,1] = angle_dif
    cond_p[1,:] = 0.
    
    skill = X_p.mean(axis=0) - cond_p
    
    spread = X_p.std(axis=0)
    
    
    sp_out[0] = skill
    
    sp_out[1] = spread


        
    return sp_out
