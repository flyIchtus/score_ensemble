"""


AROME-specific version of skill_spread

"""

import properscoring as ps
import numpy as np
import wind_comp as wc
import copy
def brier_score(cond, X, T_brier, ff_brier):
    """
    
    Inputs :
        
        X : N x C x H x W array with N samples
        
        cond : C x H x W array observation
        
    Returns :
        
        brier score  : C x H x W array containing the result
    
    """
    
    
    #cond = np.expand_dims(cond, axis = 0)
    #print(np.nanmax(cond[2]), X[:,2].max())
    N, C, H, W  = X.shape
    
    X[:,0], X[:,1] = wc.computeWindDir(X[:,0], X[:,1])
    angle_dif = wc.angle_diff(X[:,1], cond[1])

    #T_brier = 270.
    #ff_brier = 2.

    X_brier = np.zeros((C,N,H,W))
    O_brier = np.zeros((C,H,W))
    O_brier[:] = np.nan
    
    """
    Converting forecasts and observation
    """

    X_brier[0, X[:,0] > ff_brier] = 1.0
    X_brier[2, X[:,2] > T_brier] = 1.0

    X_brier_prob = X_brier.sum(axis = 1) / N
    O_brier[0, cond[0] > ff_brier] = 1
    O_brier[2, cond[2] > T_brier] = 1

    brier = ps.crps_ensemble(O_brier, X_brier_prob)
    
    for i in range(H):
        
        for j in range(W):
            
            print(O_brier[0,i,j], X_brier_prob[0,i,j], brier[0,i,j], O_brier[2,i,j], X_brier_prob[2,i,j], brier[2,i,j])   

    #for i in range(H):
        
        #for j in range(W):
            
            #print(cond[1,i,j], X[0,1,i,j], angle_dif[0,i,j],)   



    
    
    #print("Mean CRPS Ens", np.nanmean(crps[2]), np.nanmean(crps[1]), np.nanmean(crps[0]))
    
        
        
    return
