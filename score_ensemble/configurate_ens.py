#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:17:43 2022

@author: brochetc

metrics computation configuration tools

"""
import argparse
from score_ensemble.evaluation_backend_ens import var_dict
import os

def str2list(li):
    if type(li)==list:
        li2=li
        return li2
    elif type(li)==str:
        
        if ', ' in li :
            li2=li[1:-1].split(', ')
        else :
            li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

def str2tupleList(li) :
    if type(li)==list:
        li2=li
        return li2
    elif type(li)==str:
        if '),(' in li :
            li2 = li[1:-1].split('),(')
        elif '), (' in li :
            li2=li[1:-1].split('), (')
        else :
            raise ValueError("li argument is not splittable \
                             due to irregular tuple, tuple spelling")
        li3 = []
        for i, a in enumerate(li2) :
            if i==0 :
                li3.append('_'.join(a[1:].split(',')))
            elif i==len(li2)-1:
                li3.append('_'.join(a[:-1].split(',')))
            else :
                li3.append('_'.join(a.split(',')))
        return li3
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))
        
        
def retrieve_domain_parameters(path, instance_num):
    
    CI = [78,206,55,183]
    var_names = ['u','v','t2m']
    
    try :
        with open(path+'ReadMe_'+str(instance_num)+'.txt', 'r') as f :
            li=f.readlines()
            for line in li:
                if "crop_indexes" in line :
                    CI=[int(c) for c in str2list(line[15:-1])]
                    print(CI)
                if "var_names" in line :
                    var_names = [v[1:-1] for v in str2list(line[12:-1])]
            print('variables', var_names)
            f.close()
            try :
                var_real_indices = [var_dict[v] for v in var_names]
            except NameError :
                raise NameError('Variable names not found in configuration file')
            
            try :
                print(CI)
            except UnboundLocalError :
                CI = [78,206,55,183]
                
    except FileNotFoundError :
        pass
           
    return CI, var_names

def getAndNameDirs(root_expe_path):
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--expe_name', type = str, help = 'Set of experiments to dig in.', default = 'W_plus')
    parser.add_argument('--parameter_sets', type = str2tupleList, help = 'Set of numerical parameters (as tuples)', default = [])
    parser.add_argument('--instance_num', type = str2list, help = 'Instances of experiment to dig in', default = [])
    parser.add_argument('--variables', type = str2list, help = 'List of subset of variables to compute metrics on', default =[])
    
    multi_config=parser.parse_args()
    
    names=[]
    short_names=[]
    list_steps=[]
    
    nn = root_expe_path + str(multi_config.expe_name) + '/'
    print(type(multi_config.instance_num), 'instance_num')
    print(type(multi_config.parameter_sets), 'param_sets') 
    if multi_config.instance_num==[] :
        
        if multi_config.parameter_sets==[] :
            names.append(nn)
            short_names.append('00')
            list_steps.append([0])
        
        for par_name in multi_config.parameter_sets :
            names.append(nn + par_name + '/')
            
            short_names.append('par_name')
            list_steps.append([0])
    else :
        if multi_config.parameter_sets==[] :
                names.append(nn)
                short_names.append('00')
                list_steps.append([0])
                
        for par_name in multi_config.parameter_sets :
                        
            for instance in multi_config.instance_num :
                names.append(nn + par_name + '/' + 'Instance_{}/'.format(instance))
                short_names.append('Instance_{}_{}'.format(instance, par_name))
                list_steps.append([0])
                    
    data_dir_names, log_dir_names = [f+'samples/' for f in names], [f+'log/' for f in names]
    
    multi_config.data_dir_names = data_dir_names
    multi_config.log_dir_names = log_dir_names
    multi_config.short_names = short_names
    multi_config.list_steps = list_steps
    
    multi_config.length = len(data_dir_names)
    
    return multi_config

def select_Config(multi_config, index):
    """
    Select the configuration of a multi_config object corresponding to the given index
    and return it in an autonomous Namespace object
    
    Inputs :
        multi_config : argparse.Namespace object as returned by getAndNameDirs
        
        index : int
    
    Returns :
        
        config : argparse.Namespace object
    
    """    
    
    insts = len(multi_config.instance_num)

    config = argparse.Namespace() # building autonomous configuration
    
    config.data_dir_f = multi_config.data_dir_names[index]
    config.log_dir = multi_config.log_dir_names[index]
    config.steps = multi_config.list_steps[index]
    
    config.short_name = multi_config.short_names[index]
    
    if insts!=0:
        instance_index = index%insts
        config.instance_num = multi_config.instance_num[instance_index]
    
    else :
        config.instance_num = 0
    
    if len(multi_config.parameter_sets)==0 :
        config.params = '0_0'
    else :
        config.params = multi_config.parameter_sets[index]
    
    config.variables = multi_config.variables ## assuming same subset of variables for each experiment, by construction
        
    return config

class Experiment():
    """
    
    Define an "experiment" on the basis of the outputted config by select_Config
    
    This the base class to manipulate data from the experiment.
    
    It should be used exclusively as a set of informations easily regrouped in an abstract class.
    
    """
    def __init__(self, expe_config):
        
        self.data_dir_f = expe_config.data_dir_f
        self.log_dir = expe_config.log_dir
        
        if not(os.path.exists(self.log_dir)) :
            os.mkdir(self.log_dir)
        
        self.expe_dir = self.log_dir[:-4]
            
        self.steps = expe_config.steps
        
        self.instance_num = expe_config.instance_num
        
        
        ###### variable indices selection : unchanged if subset is [], else selected
        
        indices = retrieve_domain_parameters(self.expe_dir, self.instance_num)
        
        self.CI, self.var_names = indices
        
        ########### Subset selection #######
        
        var_dict_fake = { v : i for i, v in enumerate(self.var_names)} # assuming variables are ordered !
        
        self.VI_f = list(var_dict_fake.values()) # warning, special object if not modified
                
        assert set(expe_config.variables) <= set(self.var_names)
        
        if set(expe_config.variables) != set(self.var_names) \
        and not len(expe_config.variables)==0 :            
            
            self.VI_f = [var_dict_fake[v] for v in expe_config.variables ]
            
        ##### final setting of variable indices
            
        self.VI = [var_dict[v] for v in expe_config.variables]
        
        
        
        self.var_names = expe_config.variables
        
        print('Indices of selected variables in samples (real/fake) :',
              self.VI, self.VI_f)
        
        assert len(self.VI)==len(self.VI_f) # sanity check
       
        
        
        
    def __print__(self) :
    
        print("Fake data directory {}".format(self.data_dir_f))
        print("Log directory {}".format(self.log_dir))
        print("Experiment directory {}".format(self.expe_dir))
        print("Instance num {}".format(self.instance_num))
        print("Step list : ", self.steps)
        print("Crop indices", self.CI)
        print("Var names" , self.var_names)
        print("Var indices", self.VI)
