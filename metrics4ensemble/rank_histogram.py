"""


AROME-specific version of skill_spread

"""

import numpy as np
import wind_comp as wc
import copy


def rank_histo(cond, X):
    """
    
    Inputs :
        
        X : N x C x H x W array with N samples
        
        cond : C x H x W array observation
        
    Returns :
        
        bins (C, 121) max number of members in the ensemble  
    
    """
    
    N, C, H, W  = X.shape
    
    
    X_p = copy.deepcopy(X)
    cond_p = copy.deepcopy(cond)    
    X_p[:,0], X_p[:,1] = wc.computeWindDir(X_p[:,0], X_p[:,1])
    angle_dif = wc.angle_diff(X_p[:,1], cond_p[1])
    X_p[:,1] = angle_dif
    cond_p[1,~np.isnan(cond_p[1])] = 0.
    
    bins = np.zeros((C, 121)) # 121 since N=120 is the biggest ensemble... Better way to do this 
    for i in range(C):
        
        cond_var = copy.deepcopy(cond_p[i])
        X_var = copy.deepcopy(X_p[:,i])
        
        obs = cond_var[ ~np.isnan(cond_var)]
        ens = X_var[:, ~np.isnan(cond_var)]
        #print(obs.shape, ens.shape, i)
        

    
        for j in range(obs.shape[0]):
        
            ens_sort = ens[:,j]
            ens_sort.sort()
            ens_sort = np.concatenate((ens_sort, [9999999.]))
            out = np.where((ens_sort < obs[j]), True, False)
            bins[i, np.argmax(out==False)] +=1 
            #print(ens_sort, np.argmax(out==False), obs[j])
            

        
    return bins
