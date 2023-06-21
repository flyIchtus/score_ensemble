"""


AROME-specific version of skill_spread

"""

import properscoring as ps
import numpy as np
import wind_comp as wc
import copy

def brier_score(cond, X, parameters ):
    """
    
    Inputs :
        
        X : N x C x H x W array with N samples
        
        cond : C x H x W array observation
        
    Returns :
        
        brier score  : N_brier x C x H x W array containing the result N_brier number of thresholds
    
    """
    N, C, H, W  = X.shape
    
    N_brier = parameters.shape[1]
    
    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond)    
    X_p[:,0], X_p[:,1] = wc.computeWindDir(X_p[:,0], X_p[:,1])
    angle_dif = wc.angle_diff(X_p[:,1], cond_p[1])
    
    brier = np.zeros((N_brier,  C, H, W))
    
    
    for i in range(parameters.shape[1]):    

    
        T_brier = parameters[1, i]
        ff_brier = parameters[0, i]
    
        X_brier = np.zeros((C,N,H,W))
        O_brier = np.zeros((C,H,W))
        O_brier[:] = np.nan
        
        """
        Converting forecasts and observation
        """
    
        X_brier[0, X_p[:,0] > ff_brier] = 1.0
        X_brier[2, X_p[:,2] > T_brier] = 1.0
    
        X_brier_prob = X_brier.sum(axis = 1) / N
        O_brier[0, cond[0] > ff_brier] = 1
        O_brier[2, cond[2] > T_brier] = 1
        O_brier[0, cond[0] < ff_brier] = 0
        O_brier[2, cond[2] < T_brier] = 0
        
        brier[i] = ps.brier_score(O_brier, X_brier_prob)
        #if (i == 1):
            #for k in range(H):
        
                #for z in range(W):
            
                    #print(X[:,0,k,z].max(),cond[0,k,z], O_brier[0,k,z], X_brier_prob[0,k,z], brier[i,0,k,z])   
                    #print(X[:,2,k,z].max(), cond[2,k,z], O_brier[2,k,z], X_brier_prob[2,k,z], brier[i,2,k,z])   


    #print("Mean CRPS Ens", np.nanmean(crps[2]), np.nanmean(crps[1]), np.nanmean(crps[0]))
    
        
        
    return brier
