#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:42:42 2022

@author: brochetc


General metrics


"""


import numpy as np
import properscoring as ps


############################ General simple metrics ###########################

def skill_spread(obs) :
    
    
    """
    observation has to be one of the observation files from the database with name YYYYMMDD_LT.npy
    
    *** Indices of H and W where observation is available should be determined somehow...
    
    *** It is probably necessary to have a map of the domain with the latitute at the center of each "pixel"
    
    """
    
    """
    This is prototypical but it should probably look like this:
    """
    #obs = np.load('/scratch/mrmn/moldovang/obs_SE_reduced/obs20200615_0.npy')
    
    """
    Sorting the observations by longitude in order to check for duplicates
    
    """
    ind = np.argsort(obs[:,0])
    obs = obs[ind]
    
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
    
    """
    
    Pixel to lat_lon equivalence
    """
    for i in range(N_obs):
        if (obs[i,0] > Lon_min and obs[i,0] < Lon_max) and (obs[i,1] > Lat_min and obs[i,1] < Lat_max):
            
            indice_lon = np.floor((obs[i,0]-Lon_min)/dlon)
            indice_lat = np.floor((obs[i,1]-Lat_min)/dlat)
            indices_obs.append([indice_lat, indice_lon])
            obs_reduced.append(obs[i])
            
    
    indices_obs = np.array(indices_obs, dtype = 'int')
    obs_reduced = np.array(obs_reduced, dtype = 'float32')
    
    len_obs_reduced = obs_reduced.shape[0]
    #print( obs_reduced[:,4])
    """
    averaging duplicates
    """
    obs_r_clean =[]
    indices_o_clean = []
    j = 0
    sum_measurements = np.zeros((3))
    
    for i in range(len_obs_reduced):
        if (i == j): 
            sum_measurements = sum_measurements + obs_reduced[i, 2::]
            j=i+1
            #print("observation before", obs_reduced[i, 2::], i)
            if i != len_obs_reduced - 1 : ## last element....
                #print(i, j)
                while (indices_obs[i,0] == indices_obs[j,0] and indices_obs[i,1] == indices_obs[j,1]):
                    sum_measurements = sum_measurements + obs_reduced[j, 2::]
                    #print(i, j, indices_obs[i], indices_obs[j])
                    
                    #print("copy!!!!!",j, obs_reduced[j, 2::])
                    j=j+1
                    #print(i, j, indices_obs[i], indices_obs[j])
                
            observation = sum_measurements/(j-i)
            #print("observations after", observation)
            sum_measurements = np.zeros((3))
            
            #print(observation)
            
            obs_r_clean.append(observation)
            indices_o_clean.append(indices_obs[i])
            
        
    indices_o_clean = np.array(indices_o_clean, dtype = 'int')
    obs_r_clean = np.array(obs_r_clean, dtype = 'float32')    
    #print(np.unique(indices_obs, return_counts=True))       
    # for i in range(size):
    print(obs_r_clean.shape)
    """
    Creating the observation matrix with the same shape as the fake/real ensemble
    
    The strategy is to put NaN everywhere where observation is missing or where the observation is corrupted
    """
    Ens_observation = np.empty((3, 128, 128))
    Ens_observation[:] = np.nan
    
    Ens_observation[0,indices_o_clean[:,0],indices_o_clean[:,1]] = obs_r_clean[:,2] #
    Ens_observation[1, indices_o_clean[:,0], indices_o_clean[:,1]] = obs_r_clean[:,1] #
    Ens_observation[2, indices_o_clean[:,0], indices_o_clean[:,1]] = obs_r_clean[:,0] #
    
    Ens_observation[Ens_observation>1000.] =np.nan ## filtering missing readings
    Ens_observation[1][Ens_observation[0]<2.] =np.nan ## filtering dd when ff<2m/s
    
    for i in range(128):
        for j in range(128):
            
            print(Ens_observation[0,i,j], Ens_observation[1,i,j], Ens_observation[2,i,j])

    
    return Ens_observation


