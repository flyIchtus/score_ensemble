#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:31:58 2023

@author: brochetc

Executable file to use score_ensemble
"""

import score_ensemble.evaluation_frontend_ens as frontend
from score_ensemble.configurate_ens import getAndNameDirs, select_Config
import numpy as np


original_data_dir='/scratch/mrmn/moldovang/'


root_expe_path = '/scratch/mrmn/moldovang/tests_CGAN/'
    
if __name__=="__main__":
    

    
    configuration_set=getAndNameDirs(root_expe_path)
    print(configuration_set)
    N_samples = 592   
    
    program = {i :(1,N_samples) for i in range(1)}   # program={i :(1,N_samples) for i in range(1)}

    distance_metrics_list = ['bias_ensemble','rank_histogram', 'skill_spread', 'brier_score', 'ensemble_crps']#, "brier_score","entropy"]
    #distance_metrics_list = ["entropy"]
    standalone_metrics_list = ["quantiles"]#, "variance"]
    
    parameters = np.zeros((2,6))
    parameters[0] = [5., 7.5, 10., 12.5, 15., 17.5] #treshold for brier scores for wind module
    parameters[1] = [278.15, 283.15, 288.15, 293.15, 297.15, 303.15] #treshold for brier scores for temperature
    #parameters[0] = [3., 4., 5., 6., 7., 8] #treshold for brier scores for wind module
    #parameters[1] = [278.15, 281.15, 283.15, 285.15, 287.15, 289.15] #treshold for brier scores for temperature
    sbsample = configuration_set.subsample
    
    for ind in range(configuration_set.length):
        
        expe_config = select_Config(configuration_set, ind)
         
        try :
            
            
            mC = frontend.EnsembleMetricsCalculator(expe_config, 'distance_metrics')
            
            #mC.estimation(standalone_metrics_list, program, standalone=True, parallel=True, subsample=16)
            mC.estimation(distance_metrics_list, program, standalone=False, parallel=False,
                          observation = True, subsample=sbsample, parameters = parameters)
            
            mC = frontend.EnsembleMetricsCalculator(expe_config, 'rel_diagram')
            
            mC.estimation(['rel_diagram'], program, standalone=False, parallel=False,
                          observation = True, subsample=sbsample, parameters = parameters)    
                          
            
           
        except (FileNotFoundError) :
            print('File Not found  for {} ! This can be due to either \
                  inexistent experiment set, missing ReadMe file,\
                  or missing data file. Continuing !'.format(expe_config.data_dir_f))
            




