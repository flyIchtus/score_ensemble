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


original_data_dir='/scratch/mrmn/moldovang/' # pas important


root_expe_path = '/scratch/mrmn/moldovang/tests_CGAN/' # Le ensemble conditionés se trouve ici
    
if __name__=="__main__":
    

    
    configuration_set=getAndNameDirs(root_expe_path)
    print(configuration_set)
    N_samples = 300 # nombre d'ensemble sur lequels on va calculer nos score (Nous on a 15 (genSemble) dans normal)

    N_runs = 15 # nombre d'offset sur lequel on calcule les score (echeances) par date. Pour chaque simulation on a 45 offset (echeance)
    dh = 3 # echeance

    
    program = {i :(1,N_samples) for i in range(1)}   # program={i :(1,N_samples) for i in range(1)}

    distance_metrics_list = ['bias_ensemble','rank_histogram', 'skill_spread', 'brier_score', 'ensemble_crps', 'rel_diagram']#, "brier_score","entropy"]
    #distance_metrics_list = ['brier_score']#, "brier_score","entropy"]

    #distance_metrics_list = ['skill_spread']#, "brier_score","entropy"]
        
    
    #distance_metrics_list = ['ensemble_crps']#, "brier_score","entropy"]
    #distance_metrics_list = ["mse", "bias"]
    #standalone_metrics_list = ["quantiles"]#, "variance"]
    
    parameters = np.zeros((2,6))
    parameters[0] = [1.39, 2.78, 4.17, 5.56, 8.33, 11.11] #treshold for brier scores for wind module [m/s]
    parameters[1] = [278.15, 283.15, 288.15, 293.15, 297.15, 303.15] #treshold for brier scores for temperature [K]

    sbsample = configuration_set.subsample #membres d'ensembles generés. ça peut etre un multiple du nombre de membres de la base de données AROME
    debiasing = configuration_set.debiasing # 
    num_proc = configuration_set.num_proc # paralelization
    data_dir_real = configuration_set.data_dir_real # Path ou il ira chercher IS_XXXXXXXXX la bdd AROME
    data_dir_obs = configuration_set.data_dir_obs # Path ou il ira chercher les obs!
    inv_step = configuration_set.inv_step
    conditioning_members = configuration_set.conditioning_members
    N_ensemble = configuration_set.N_ensemble
    for ind in range(configuration_set.length):
        
        expe_config = select_Config(configuration_set, ind)
         
        #try :
            
        
        mC = frontend.EnsembleMetricsCalculator(expe_config, 'distance_metrics')
        
        #mC.estimation(standalone_metrics_list, program, standalone=True, parallel=True, subsample=16)
        mC.estimation(distance_metrics_list, program, standalone=False, parallel=False,
                      observation = True, subsample=sbsample, parameters = parameters, N_runs=N_runs, dh=dh, debiasing = debiasing,
                      num_proc=num_proc, inv_step=inv_step, conditioning_members=conditioning_members,N_ensemble=N_ensemble,
                      data_dir_real=data_dir_real, data_dir_obs = data_dir_obs)
        
        #mC = frontend.EnsembleMetricsCalculator(expe_config, 'rel_diagram')
        
        #mC.estimation(['rel_diagram'], program, standalone=False, parallel=False,
        #              observation = True, subsample=sbsample, parameters = parameters, N_runs=N_runs, dh=dh)    
                      
        
       
        #except (FileNotFoundError) :
        #    print('File Not found  for {} ! This can be due to either \
        #          inexistent experiment set, missing ReadMe file,\
        #         or missing data file. Continuing !'.format(expe_config.data_dir_f))
            




