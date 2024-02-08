#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:04:19 2022

@author: brochetc


Metrics computation automation

"""


import pickle
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
import pandas as pd
import copy
import glob

from score_ensemble.configurate_ens import Experiment
import score_ensemble.evaluation_backend_ens as backend
import metrics4ensemble as metrics



########### standard parameters #####

var_dict = backend.var_dict 
#data_dir_real = backend.data_dir_real
#data_dir_obs = backend.data_dir_obs

#####################################

   
class EnsembleMetricsCalculator(Experiment) :
    
    def __init__(self, expe_config, add_name) :
        super().__init__(expe_config)
        
        
        self.add_name = add_name
    
    def __print__(self):
        super().__print__()
    ###########################################################################
    ######################### Main class method ###############################
    ###########################################################################
    
    
    def estimation(self, metrics_list, program, subsample=16, parallel=False, standalone=False,
                   same=False, real=False, observation = False, parameters = None, N_runs=None, dh=None, debiasing = False,
                   num_proc=8,inv_step = None, conditioning_members=None, N_ensemble=None, data_dir_real=None, data_dir_obs=None) :
        
        """
        
        estimate all metrics contained in metrics_list on training runs
        using specific strategies
                       -> parallel or sequential estimation
                       -> distance metrics or standalone metrics
                       -> on real samples only (if distance metrics)
                       
        Inputs :
            
            metrics_list : list, the list of metrics to be computed
            
            program : dict of shape {int : (int, int)}
                      contains all the informations about sample numbers and number of repeats
                      #### WARNING : in this case, each sample is supposed to represent a given ensemble
                      
                      keys index the repeats
                      
                      values[0] index the type of dataset manipulation
                      (either dividing the same dataset into parts, or selecting only one portion)
                      
                      values[1] indicate the number of samples to use in the computation
                      
                      Note : -> for tests on training dynamics, only 1 repeat is assumed
                                  (at the moment)
                             -> for tests on self-distances on real datasets,
                                many repeats are possible (to account for test variance
                                or test different sampling sizes)
            
            parallel, standalone, real : bool, the flags defining the estimation
                                         strategy
        
        Returns :
            
            None
            
            dumps the results in a pickle file
        
        """
        
        ########### sanity checks
        
        if standalone and not parallel :
            raise(ValueError, 'Estimation for standalone metric should be done in parallel')
        
        if standalone :
            
            assert set(metrics_list) <= metrics.standalone_metrics
        
        else :
            
            assert set(metrics_list) <= metrics.distance_metrics
        
        for metric in metrics_list :
            assert hasattr(metrics, metric)
            
        ########################
        
        self.program = program
        self.parameters = parameters
        self.N_runs = N_runs
        self.dh = dh
        self.debiasing = debiasing
        self.num_proc = num_proc
        self.data_dir_real = data_dir_real
        self.data_dir_obs = data_dir_obs
        self.inv_step = inv_step
        self.conditioning_members = conditioning_members
        self.N_ensemble = N_ensemble
        
        ################### SOME CUISINE TO GET DATE LIST AND OBSERVATION DATE LIST #########
        df0 = pd.read_csv(self.data_dir_real + 'Large_lt_test_labels.csv')
        
        df1 = pd.read_csv(self.data_dir_real + 'Large_lt_test_labels.csv')
        
        List_dates_unique = df0["Date"].unique().tolist()
        List_dates_inv = df1["Date"].unique().tolist()
        
        List_dates_inv.remove('2021-10-29T21:00:00Z') #31.10.2021 obs missing
        List_dates_inv.remove('2021-10-30T21:00:00Z')
        List_dates_unique.remove('2021-10-29T21:00:00Z') #31.10.2021 obs missing
        List_dates_unique.remove('2021-10-30T21:00:00Z')
        List_dates_inv_org = copy.deepcopy(List_dates_inv)
        #List_dates_unique.sort()
        #List_dates_inv_org.sort()
        #List_dates_inv.sort()
        
        for i in range(len(List_dates_unique)):
            List_dates_unique[i]=List_dates_unique[i].replace("T21:00:00Z","")
            List_dates_unique[i]=List_dates_unique[i].replace("-","")
        
        for i in range(len(List_dates_inv)):
            List_dates_inv[i]=List_dates_inv[i].replace("T21:00:00Z","")
            
        ############### Putting all the available observation dates in a list
        
        fl_obs = glob.glob(data_dir_obs+"/obs*")
        
        len_fl_obs = len(fl_obs)
        
        for i in range(len_fl_obs):
            fl_obs[i] = fl_obs[i].replace(data_dir_obs, "")
            fl_obs[i] = fl_obs[i].replace(".npy", "")
            fl_obs[i] = fl_obs[i].replace("obs", "")
            
            fl_obs[i] = fl_obs[i][0:8]
            #print(fl_obs[i])
        
        fl_obs=list(set(fl_obs))
        fl_obs.append('20211031') ########## ATTENTION ARTIFICIALLY ADDING THIS DATES BECAUSE IT DOES NOT EXIST IN THE DB        

        
        fl_obs.sort()


        ################### SOME CUISINE TO GET DATE LIST AND OBSERVATION DATE LIST #########
        
        self.fl_obs = fl_obs
        self.List_dates_inv = List_dates_inv
        self.List_dates_inv_org = List_dates_inv_org
        self.List_dates_unique = List_dates_unique
        print('Subsample', subsample)
        
        option = 'real' if real else 'fake'
            
        if parallel :
            
            if standalone :
                
                name='_standalone_metrics_'
                    
                func = lambda m_list : self.parallelEstimation_standAlone(m_list, 
                                                                          subsample = subsample,
                                                                          option=option)
                
            else :
                
                name='_distance_metrics_'
                
                if real :
                    func = lambda m : self.parallelEstimation_sameVSsame(m, 
                                                                         subsample = subsample, option='real')
                else :
                    func = lambda m : self.parallelEstimation_realVSfake(m, 
                                                                         subsample = subsample)
        else :
            
            name='_distance_metrics_'
            
            if same :
                
                func = lambda m : self.sequentialEstimation_sameVSsame(m, 
                                                                       subsample = subsample, option = option)
                
            else :
                func = self.sequentialEstimation_realVSfake
                
            if observation :
                
                func = lambda m : self.Estimation_modelvsobs(m, 
                                                                       subsample = subsample, option = option)
                
        results = func(metrics_list)
            
        N_samples_set = [self.program[i][1] for i in range(len(program))]
            
        N_samples_name = '_'.join([str(n) for n in N_samples_set])
        
        if real : 
            
            temp_log_dir = self.log_dir
            
            self.log_dir = backend.data_dir_real
        
        #dumpfile = self.log_dir + self.add_name+name + str(N_samples_name)+'.p'
        dumpfile = self.log_dir + self.add_name+name + str(N_samples_name)
        if real : 
            
            self.log_dir = temp_log_dir


        for i in range(len(metrics_list)):
            np.save(dumpfile+'_'+metrics_list[i] +'_' +str(self.inv_step) +
                    '_' + str(self.conditioning_members) + '_' + str(self.N_ensemble) + '.npy',
                    results[0][metrics_list[i]])  ### ATTENTION

        #print(dumpfile, metrics_list, results)
        #pickle.dump(results, open(dumpfile, 'wb'))
        
        
    
    ###########################################################################
    ############################   Estimation strategies ######################
    ###########################################################################
    
    
    
    def parallelEstimation_realVSfake(self, metrics_list, subsample=16):
        
        """
        
        makes a list of datasets with each item of self.steps
        and use multiple processes to evaluate the metric in parallel on each
        item.
        
        The metric must be a distance metric and the data should be real / fake
        
        Inputs : 
            
            metric : str, the metric to evaluate
            
            subsample : tuple or int the number of members to subsample
                        either from both the same (int) of from (real, fake)
            
        Returns :
            
            res : ndarray, the results array (precise shape defined by the metric)
        
        """
        
        RES = {}
        
        for step in self.steps: # for the moment, not the point using step but this can come
            
            print('Step', step)
            
            dataset_f, indexList = backend.build_datasets(self.data_dir_f, self.program,  N_runs=self.N_runs, dh=self.dh)
            
            dataset_r, _ = backend.build_datasets(self.data_dir_real, self.program,
                                               option = 'real',
                                               indexList = indexList,  N_runs=self.N_runs, dh=self.dh) # taking the 'same ensemble' for fake and real
            
           
            data_list = []
            
            for i0 in self.program.keys():
                                
                data_r = dataset_r[i0]
                
                N_samples = self.program[i0][1]
                
                #getting files to analyze from fake dataset
                data_f = dataset_f[i0]
                
              
                data_list.append((metrics_list, {'real': data_r,'fake': data_f},\
                                  N_samples, N_samples,\
                                  self.VI, self.VI_f, self.CI, i0, subsample, self.parameters, self.N_runs, self.dh, self.debiasing))

            with Pool(self.num_proc) as p :
                res = p.map(backend.eval_distance_metrics, data_list)
                    
                
            ## some cuisine to produce a rightly formatted dictionary
            
            ind_list=[]
            d_res = defaultdict(list)
            
            for res_index in res :
                index = res_index[1]
                res0 = res_index[0]
                for k, v in res0.items():                    
                    d_res[k].append(v)
                ind_list.append(index)
            
            for k in d_res.keys():
                d_res[k]= [x for _,x in sorted(zip(ind_list, d_res[k]))]
            
            res = { k : np.concatenate([np.expand_dims(v[i], axis=0) \
                                        for i in range(len(self.program.keys()))], axis=0).squeeze()
                                        for k,v in d_res.items()}
            RES[step] = res
            
        if step==0 :

            return res
        return RES


    def Estimation_modelvsobs(self, metrics_list, subsample=16, option = 'real'):
        
        """
        
        makes a list of datasets with each item of self.steps
        and use multiple processes to evaluate the metric in parallel on each
        item.
        
        The metric must be a distance metric and the data should be real / fake
        
        Inputs : 
            
            metric : str, the metric to evaluate
            
            subsample : tuple or int the number of members to subsample
                        either from both the same (int) of from (real, fake)
            
        Returns :
            
            res : ndarray, the results array (precise shape defined by the metric)
        
        """

        RES = {}
        
        for step in self.steps: # for the moment, not the point using step but this can come
            
            print('Step', step)
            
            dataset_f, indexList = backend.build_datasets(self.data_dir_f, self.program, N_runs=self.N_runs, dh=self.dh,
                                                          data_dir_real=self.data_dir_real, List_dates_inv=self.List_dates_inv, List_dates_inv_org=self.List_dates_inv_org,
                                                          inv_step=self.inv_step, conditioning_members=self.conditioning_members, N_ensemble=self.N_ensemble)
            
            dataset_o, _ = backend.build_datasets(self.data_dir_obs, self.program,
                                               option = 'observation',
                                               indexList = indexList, N_runs=self.N_runs, dh=self.dh,
                                               data_dir_real=self.data_dir_real, List_dates_inv=self.List_dates_inv, List_dates_inv_org=self.List_dates_inv_org,
                                               inv_step=self.inv_step, conditioning_members=self.conditioning_members, N_ensemble=self.N_ensemble) 
            dataset_r, _ = backend.build_datasets(self.data_dir_real, self.program,
                                               option = 'real',
                                               indexList = indexList, N_runs=self.N_runs, dh=self.dh,
                                               data_dir_real=self.data_dir_real, List_dates_inv=self.List_dates_inv, List_dates_inv_org=self.List_dates_inv_org,
                                               inv_step=self.inv_step, conditioning_members=self.conditioning_members, N_ensemble=self.N_ensemble) 
            data_list = []
            for i0 in self.program.keys():
                
                data_o = dataset_o[i0]
                
                N_samples = self.program[i0][1]
                
                #getting files to analyze from fake dataset
                data_f = dataset_f[i0]
                data_r = dataset_r[i0]
                
              
                data_list.append((metrics_list, {'obs': data_o,'fake': data_f,'real': data_r},\
                                  N_samples, N_samples,\
                                  self.VI, self.VI_f, self.CI, i0, subsample, self.parameters, self.N_runs, self.dh, self.debiasing, self.conditioning_members,
                                  self.data_dir_real, self.data_dir_obs,self.List_dates_inv, self.List_dates_inv_org, self.List_dates_unique,self.fl_obs))
            #res = []
            #with Pool(num_proc) as p :
            #    res = p.map(backend.eval_distance_metrics, data_list)
                for k,i0 in enumerate(self.program.keys()):
                    res = backend.eval_distance_metrics(data_list[k])
            #res.append(backend.eval_distance_metrics(data_list))
                
            ## some cuisine to produce a rightly formatted dictionary
            ind_list=[]
            d_res = defaultdict(list)
            
            #for res_index in res :
            #    index = res_index[1]
            #    res0 = res_index[0]
            #    for k, v in res0.items():                    
            #        d_res[k].append(v)
            #    ind_list.append(index)
            
            #for k in d_res.keys():
            #    d_res[k]= [x for _,x in sorted(zip(ind_list, d_res[k]))]
            
            #res = { k : np.concatenate([np.expand_dims(v[i], axis=0) \
            #                            for i in range(len(self.program.keys()))], axis=0).squeeze()
            #                            for k,v in d_res.items()}
            RES[step] = res
            
        if step==0 :

            return res
        return RES

        
    
        
    
    def sequentialEstimation_realVSfake(self, metrics_list, subsample=16):
        
        """
        
        Iterates the evaluation of the metric on each item of self.steps
        
        The metric must be a distance metric and the data should be real / fake
        
        Inputs : 
            
            metric : str, the metric to evaluate
            
            subsample : tuple or int the number of members to subsample
                        either from both the same (int) of from (real, fake)
            
        Returns :
            
            res : ndarray, the results array (precise shape defined by the metric)
        
        """
        
        RES = {}
        
        
        for step in self.steps:
            
            print('Step', step)
        
            dataset_f, indexList = backend.build_datasets(self.data_dir_f, self.program, N_runs=self.N_runs, dh=self.dh,
                                                          data_dir_real=self.data_dir_real, List_dates_inv=self.List_dates_inv, List_dates_inv_org=self.List_dates_inv_org)
            
            dataset_r, _ = backend.build_datasets(self.data_dir_real, self.program,
                                               option = 'real',
                                               indexList = indexList, N_runs=self.N_runs, dh=self.dh,
                                               data_dir_real=self.data_dir_real, List_dates_inv=self.List_dates_inv, List_dates_inv_org=self.List_dates_inv_org) # taking the 'same ensemble' for fake and real
            
            res = []
            
            for i0 in self.program.keys() :
            
                #getting first (and only) item of the random real dataset program
                data_r = dataset_r[i0]
                
                N_samples = self.program[i0][1]
                
                #getting files to analyze from fake dataset
                
                data_f = dataset_f[i0]
                
                data = (metrics_list, {'real': data_r,'fake': data_f},\
                      N_samples, N_samples,
                      self.VI, self.VI_f, self.CI, i0, subsample,  self.parameters, self.N_runs, self.dh, self.debiasing,
                      self.data_dir_real, self.data_dir_obs,self.List_dates_inv, self.List_dates_inv_org, self.List_dates_unique,self.fl_obs)
       
                res.append(backend.eval_distance_metrics(data))
          
            ## some cuisine to produce a rightly formatted dictionary
                
            d_res = defaultdict(list)
            
            for res_index in res :
                res0 = res_index[0]
                for k, v in res0.items():
                    d_res[k].append(v)
            
            res = { k : np.concatenate([np.expand_dims(v[i], axis=0) \
                                        for i in range(len(self.program.keys()))], axis=0).squeeze() 
                                        for k,v in d_res.items()}
            
            RES[step] = res
            
        if step==0 :
            return res
        
        return RES
        
        
    def parallelEstimation_sameVSsame(self, metric, subsample=16, option = 'real'):
        
        """
        
        makes a list of datasets with each pair of real datasets contained
        in self.program.
        
        Use multiple processes to evaluate the metric in parallel on each
        item.
        
        The metric must be a distance metric and the data should be real / real
        
        Inputs : 
            
            metric : str, the metric to evaluate
            
            subsample : tuple or int the number of members to subsample
                        either from both the same (int) of from (0, 1)
            option : str, whether to apply to real or fake samples
            
        Returns :
            
            res : ndarray, the results array (precise shape defined by the metric)
        
        """
        
        if option=='real' :
            
            datasets = backend.build_datasets(self.data_dir_real, self.program)
            data_list = []         
        
            #getting the two random datasets programs
                
            for i in range(len(datasets)):
                
                N_samples = self.program[i][1]
              
                data_list.append((metric, 
                                  {'real0':datasets[i][0],'real1': datasets[i][1]},
                                  N_samples, N_samples,
                                  self.VI, self.VI, self.CI, i, subsample,  self.parameters, self.N_runs, self.dh, self.debiasing))
            
            with Pool(self.num_proc) as p :
                res = p.map(backend.eval_distance_metrics, data_list)
        
        elif option=='fake' :
        
            datasets = backend.build_datasets(self.data_dir_f, self.program)
            data_list = []         
            
            #getting the two random datasets programs
                
            for i in range(len(datasets)):
                
                N_samples = self.program[i][1]
              
                data_list.append((metric, 
                                  {'fake0':datasets[i][0],'fake1': datasets[i][1]},
                                  N_samples, N_samples,
                                  self.VI_f, self.VI_f, self.CI,i, subsample))
            
            with Pool(self.num_proc) as p :
                res = p.map(backend.eval_distance_metrics, data_list)
        ## some cuisine to produce a rightly formatted dictionary
            
        ind_list=[]
        d_res = defaultdict(list)
            
        for res_index in res :
            index = res_index[1]
            res0 = res_index[0]
            for k, v in res0.items():
                d_res[k].append(v)
            ind_list.append(index)
        
        for k in d_res.keys():
            d_res[k]= [x for _, x in sorted(zip(ind_list, d_res[k]))]
        
        res = { k : np.concatenate([np.expand_dims(v[i], axis=0) \
                                    for i in range(len(self.program.keys()))], axis=0).squeeze() 
                                    for k,v in d_res.items()}
        
        return res
        
    def sequentialEstimation_sameVSsame(self, metric, subsample=16, option = 'real'):
        
        """
        
        Iterates the evaluation of the metric on each item of pair of real datasets 
        defined in self.program.
        
        The metric must be a distance metric and the data should be real / fake
        
        Inputs : 
            
            metric : str, the metric to evaluate
            
            subsample : tuple or int the number of members to subsample
                        either from both the same (int) of from (0, 1)
            
            option : str, whether data is fake or real
            
        Returns :
            
            res : ndarray, the results array (precise shape defined by the metric)
        
        """
        
            
        #getting first (and only) item of the random real dataset program
        
        
        if option =='real':
        
            datasets = backend.build_datasets(self.data_dir_real, self.program)
                
            for i in range(len(datasets)):
                
                N_samples = self.program[i][1]
              
                data=(metric, {'real0':datasets[i][0],'real1': datasets[i][1]},
                               N_samples, N_samples,
                               self.VI, self.VI, self.CI, i, subsample, self.parameters)
            
                if i==0: res = [backend.eval_distance_metrics(data)]
                else :  
                    res.append(backend.eval_distance_metrics(data))
        
        if option=='fake' :
        
            datasets = backend.build_datasets(self.data_dir_f, self.program)

            for i in range(len(datasets)):
                
                N_samples = self.program[i][1]
              
                data = (metric, {'fake0':datasets[i][0],'fake1': datasets[i][1]},
                               N_samples, N_samples,
                               self.VI_f, self.VI_f, self.CI, i, subsample)
            
                if i==0: res = [backend.eval_distance_metrics(data)]
                else :  
                    res.append(backend.eval_distance_metrics(data))
        ## some cuisine to produce a rightly formatted dictionary
       
        d_res = defaultdict(list)
        
        for res_index in res :
            res0 = res_index[0]
            for k, v in res0.items():
                d_res[k].append(v)
        
        res = { k : np.concatenate([np.expand_dims(v[i], axis=0) \
                                    for i in range(len(self.program.keys()))], axis=0).squeeze() 
                                    for k,v in d_res.items()}
            
        return res
        
        
    def parallelEstimation_standAlone(self, metrics_list, subsample=16, option='fake'):
        
        """
        
        makes a list of datasets with each dataset contained
        in self.program (case option =real) or directly from data files 
        (case option =fake)
        
        Use multiple processes to evaluate the metric in parallel on each
        item.
        
        The metric must be a standalone metric.
        
        Inputs : 
            
            metric : str, the metric to evaluate
            
        Returns :
            
            res : ndarray, the results array (precise shape defined by the metric)
        
        """
        
        if option=='real':
            
            
            dataset_r,_ = backend.build_datasets(self.data_dir_real, self.program, option = 'real')   
            print(len(dataset_r))
            data_list = [(metrics_list, dataset_r, self.program[i][1], 
                          self.VI, self.VI, self.CI, i, option, subsample) \
                        for i, dataset in enumerate(dataset_r)]
            
            with Pool(self.num_proc) as p :
                res = p.map(backend.global_dataset_eval, data_list)
            
            ind_list=[]
            d_res = defaultdict(list)
                
            for res_index in res :
                index = res_index[1]
                res0 = res_index[0]
                for k, v in res0.items():
                    d_res[k].append(v)
                ind_list.append(index)
            
            for k in d_res.keys():
                d_res[k]= [x for _, x in sorted(zip(ind_list, d_res[k]))]
            
            res = { k : np.concatenate([np.expand_dims(v[i], axis=0) \
                                        for i in range(len(self.steps))], axis=0).squeeze() 
                   for k,v in d_res.items()}
                
            return res

        elif option=='fake' :
            
            RES = {}
            
            for j,step in enumerate(self.steps):
            
                dataset_f = backend.build_datasets(self.data_dir_f, self.program)   
                
                data_list = []
                
                for i0 in self.program.keys():
                
                    files = dataset_f[i0]
                        
                    data_list.append((metrics_list, files, self.program[i0][1], 
                                          self.VI_f, self.VI_f, self.CI, step, option,
                                          subsample))
                    
                with Pool(self.num_proc) as p :
                    res = p.map(backend.global_dataset_eval, data_list)
                    
                ind_list=[]
                d_res = defaultdict(list)
                    
                for res_index in res :
                    index = res_index[1]
                    res0 = res_index[0]
                    for k, v in res0.items():
                        d_res[k].append(v)
                    ind_list.append(index)
                    
                for k in d_res.keys():
                    d_res[k]= [x for _, x in sorted(zip(ind_list, d_res[k]))]
            
                res = { k : np.concatenate([np.expand_dims(v[i], axis=0) \
                                            for i in range(len(self.program.keys()))], axis=0).squeeze() 
                       for k,v in d_res.items()}
                    
                RES[j] = res
                    
            if j==0 :
                return res
        
            return RES

