#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:53:46 2022

@author: brochetc

metrics version 2

File include :
    
    metric2D and criterion2D APIs to be used by Trainer class from trainer_horovod
    provide a directly usable namespace from already implemented metrics

"""

import general_metrics as GM
import quantiles_metric as quant
import entropy as Entropy
import CRPS as CRPS
import mean_bias as mb
import metrics4ensemble.spectral_variance as spvar

import numpy as np
###################### standard parameters

var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' : 3 , 'orog' : 4}

var_dict_fake = {'u' : 0 , 'v' : 1, 't2m' : 2, 'orog' : 3}

vars_wo_orog = ['u', 'v', 't2m']

######################

########################### High level APIs ##################################

class metric2D_ens():
    def __init__(self, long_name, func, variables, end_shape=(3,128,128), names = ['metric']):
        
        self.long_name = long_name
        
        self.names = names # names for each of the func's output items
        
        self.func = func #should return np.array OR tensor to benefit from parallel estimation
        
        self.end_shape = end_shape # the final shape of the tensor returned by the metric 
                                   #(except for the 'batch' or 'sample' direction)
        
        self.variables = variables # variables on which the metric is applied
        
    def selectVars(self, *args) :
        
        """
        select in the input data the variables to compute metric on
        """
        
        if len(args)==2 :
            
            real_data, fake_data = args[0]
            
            VI = [var_dict[v] for v in self.variables]
            VI_f = [var_dict_fake[v] for v in self.variables]
            
            real_data = real_data[:, VI,:,:]
            print(VI,VI_f)
            fake_data = fake_data[:, VI_f,:,:]
        
            return real_data, fake_data
        
        else :
            
            return args[0]
    

    def __call__(self, *args, **kwargs):
        
        
        ########## selecting variables check #########
        try :
            select = kwargs['select']
        except KeyError :
            
            select = True
        
        ############# selection ################
    
        if select :
        
            data = self.selectVars(args)
        
        else :
            
            data = args
            
        ########### computation ################

        reliq_kwargs ={ k :v for k,v in kwargs.items() if k!='select'}
        
        if len(data) == 2:
            
            res = np.zeros((data[0].shape[0] + 2,) + self.end_shape).astype(np.float32)
            
            for i in range(data[0].shape[0]) :
                res[i] = self.func(data[0][i], data[1][i],**reliq_kwargs)
            
            res[-2], res[-1] = res[:-2].mean(axis = 0), res[:-2].std(axis = 0)
            # adding the batch mean and std at the end of the vector

                        
            return res
        
        else :
            
            print(data[0].shape)
            
            res = np.zeros((data[0].shape[0] + 2,) + self.end_shape).astype(np.float32)
            
            for i in range(data[0].shape[0]) :
                
                res[i] = self.func(data[0][i],**reliq_kwargs)
            
            res[-2], res[-1] = res[:-2].mean(axis = 0), res[:-2].std(axis = 0)
            # adding the batch mean and std at the end of the vector

            return res
   


##############################################################################
        ################## Metrics catalogue #####################
        
standalone_metrics = {"spectral_dev","spectral_var","quantiles", "variance"}

distance_metrics = {"quantile_score", "entropy", "ensemble_crps", "variance_diff", "mean_bias", "std_diff", "rel_std_diff"}


###################### Usable namespace #######################################


quantiles = metric2D_ens('Multiple Quantiles', 
                         quant.quantiles, vars_wo_orog, end_shape =(7,3,128,128) )

quantile_score = metric2D_ens('Quantile RMSE', 
                              quant.quantile_score, vars_wo_orog)

entropy = metric2D_ens('Added ensemble entropy', 
                       Entropy.ensemble_log_likelihood, vars_wo_orog)

ensemble_crps = metric2D_ens('Average crps',
                             CRPS.ensemble_crps, vars_wo_orog)


variance = metric2D_ens('Variance per variable',
                        GM.simple_variance, vars_wo_orog)

variance_diff = metric2D_ens('Variance map difference', GM.variance_diff, 
                             vars_wo_orog)

std_diff = metric2D_ens('Standard variation map difference', GM.std_diff, 
                             vars_wo_orog)


mean_bias = metric2D_ens('Average bias', mb.mean_bias, vars_wo_orog)

rel_std_diff = metric2D_ens('Relative Std change', GM.relative_std_diff, vars_wo_orog)

spectral_dev = metric2D_ens('Spectral deviation', spvar.spectrum_deviation, vars_wo_orog,  end_shape = (3,45))

spectral_var = metric2D_ens('Spectral deviation', spvar.spectrum_variance, vars_wo_orog,  end_shape = (3,45))