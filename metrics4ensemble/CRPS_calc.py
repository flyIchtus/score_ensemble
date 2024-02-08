#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:38:11 2023

@author: brochetc

AROME-specific version of CRPS

"""

import properscoring as ps
import numpy as np
import metrics4ensemble.wind_comp as wc
import copy
import CRPS.CRPS as psc
def ensemble_crps(cond, X, real_ens, debiasing = False, conditioning_members=None):
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
    real_ens_p = copy.deepcopy(real_ens)
    

    
    ##################################################################
    X_p[:,0], X_p[:,1] = wc.computeWindDir(X_p[:,0], X_p[:,1])
    real_ens_p[:,0], real_ens_p[:,1] = wc.computeWindDir(real_ens_p[:,0], real_ens_p[:,1])
    
    



    if debiasing != 'None' : 

        X_p = wc.debiasing(X_p, real_ens_p, conditioning_members, mode=debiasing)


    
    angle_dif = wc.angle_diff(X_p[:,1], cond_p[1])


    X_p[:,1] = angle_dif
    cond_p[1,~np.isnan(cond_p[1])] = 0.


    print(cond_p.shape)
    cond_p_ff = cond_p[0,~np.isnan(cond_p[0])]
    cond_p_dd = cond_p[1,~np.isnan(cond_p[1])]
    cond_p_t2m = cond_p[2,~np.isnan(cond_p[2])]
    
    X_p_ff = X_p[:,0,~np.isnan(cond_p[0])]
    X_p_dd = X_p[:,1,~np.isnan(cond_p[1])]
    X_p_t2m = X_p[:,2,~np.isnan(cond_p[2])]
    
    print(X_p_ff.shape, X_p_dd.shape, X_p_t2m.shape)
    
    crps_res = np.zeros((3,1))
    sm = 0.
    for i in range(len(cond_p_ff)):
        
        crps,fcrps,acrps = psc(X_p_ff[:,i],cond_p_ff[i]).compute()   
        sm = sm + crps
    crps_res[0] = sm / len(cond_p_ff) 
    sm = 0.
    
    for i in range(len(cond_p_dd)):
        
        crps,fcrps,acrps = psc(X_p_dd[:,i],cond_p_dd[i]).compute()   
        sm = sm + crps
    crps_res[1] = sm / len(cond_p_dd)
    sm = 0.

    for i in range(len(cond_p_t2m)):
        
        crps,fcrps,acrps = psc(X_p_t2m[:,i],cond_p_t2m[i]).compute()   
        sm = sm + crps
    crps_res[2] = sm / len(cond_p_t2m)    

    print(crps_res)
    #cond
   
    return crps_res
