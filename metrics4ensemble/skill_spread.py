#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:42:42 2022

@author: brochetc


General metrics


"""


import numpy as np


############################ General simple metrics ###########################

def skill_spread(X) :
    
    
    """
    X :  N x C H x W array
    
    obs : shape to be determined but probaably its spares C x H x W
    
    *** Indices of H and W where observation is available should be determined somehow...
    
    *** It is probably necessary to have a map of the domain with the latitute at the center of each "pixel"
    
    """
    
    """
    This is prototypical but it should probably look like this:
    """
    obs = np.load('/scratch/mrmn/moldovang/observation/obs20200615_0.npy')
    
    N_obs = obs.shape[0]
    
    Lat_min = 42.44309623430962
    Lon_min = 2.8617305976806424
    Lat_max = 45.63863319386332
    Lon_max = 6.058876003568244
    size = 128
    
    dlat = (Lat_max - Lat_min)/size
    dlon = (Lon_max - Lon_min)/size
    
    
    indices_obs = np.zeros((N_obs, 2))
    
    obs_reduced = []
    indices_obs = []
    for i in range(N_obs):
        if (obs[i,0] > Lon_min and obs[i,0] < Lon_max) and (obs[i,1] > Lat_min and obs[i,1] < Lat_max):
            
            indice_lon = np.floor((obs[i,0]-Lon_min)/dlon)
            indice_lat = np.floor((obs[i,1]-Lat_min)/dlat)
            indices_obs.append([indice_lat, indice_lon])
            obs_reduced.append(obs[i])
            
    
    indices_obs = np.array(indices_obs, dtype = 'int')
    obs_reduced = np.array(obs_reduced, dtype = 'float32')
    
    len_obs_reduced = obs_reduced.shape[0]
    for i in range(len_obs_reduced):
        
    # for i in range(size):
    #     for j in range(size):
            
    #         Lat = Lat_min + i*dlat
    #         Lon = Lon_min + j*dlon
            
    #         for k in range(N_obs):
                
    #             if (Lat < obs[k,1] and Lat + dlat >  obs[k,1]) and (Lon < obs[k,0] and Lon + dlon >  obs[k,0]):
                    
    #                 indices_obs[k] = [i, j]
                    
    #                 print(indices_obs[k], Lat, Lon)
    
    return "hehe"


