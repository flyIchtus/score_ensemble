#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:31:58 2023

@author: brochetc

Executable file to use score_ensemble
"""

import score_ensemble.evaluation_frontend_ens as frontend
from score_ensemble.configurate_ens import getAndNameDirs, select_Config


original_data_dir='/scratch/mrmn/brochetc/'


root_expe_path = '/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/Database_latent/'
    
if __name__=="__main__":
    
    configuration_set=getAndNameDirs(root_expe_path)
    
    N_samples = 10   
    
    program = {i :(1,N_samples) for i in range(1)}   # program={i :(1,N_samples) for i in range(1)}

    distance_metrics_list = ["std_diff", "mean_bias","entropy"]
    standalone_metrics_list = ["quantiles"]#, "variance"]
    
    for ind in range(configuration_set.length):
        
        expe_config = select_Config(configuration_set, ind)
         
        try :
            
            mC = frontend.EnsembleMetricsCalculator(expe_config, 'qtiles')
            
            mC.estimation(standalone_metrics_list, program, standalone=True, parallel=True, subsample=120)
           
        except (FileNotFoundError) :
            print('File Not found  for {} ! This can be due to either \
                  inexistent experiment set, missing ReadMe file,\
                  or missing data file. Continuing !'.format(expe_config.data_dir_f))
            




