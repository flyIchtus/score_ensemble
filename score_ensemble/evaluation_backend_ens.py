#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:54:05 2022

@author: brochetc


"""
import numpy as np
import metrics4ensemble as metrics
import score_ensemble.useful_funcs as oc
import random
import pandas as pd
import glob
import copy



############## ATTENTION HARD CODING AHEAD###############
########### standard parameters #####

#num_proc = 8
var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4} # do not touch unless
                                                          # you know what u are doing

#data_dir_real = '/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/'
#data_dir_real = '/scratch/work/brochetc/datasets/IS_1_1.0_0_0_0_0_0_256_large_lt_done/'
#data_dir_obs = '/scratch/work/brochetc/datasets/obs_full_npy/'

#data_dir_inv = '/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_large_lt_done/'

################################ This is done to obtain the dates used for the forecast and the observation

# df0 = pd.read_csv(data_dir_real + 'Large_lt_test_labels.csv')

# df1 = pd.read_csv(data_dir_real + 'Large_lt_test_labels.csv')

# List_dates_unique = df0["Date"].unique().tolist()
# List_dates_inv = df1["Date"].unique().tolist()

# List_dates_inv.remove('2021-10-29T21:00:00Z') #31.10.2021 obs missing
# List_dates_inv.remove('2021-10-30T21:00:00Z')
# List_dates_unique.remove('2021-10-29T21:00:00Z') #31.10.2021 obs missing
# List_dates_unique.remove('2021-10-30T21:00:00Z')

# List_dates_inv_org = copy.deepcopy(List_dates_inv)
# List_dates_unique.sort()
# List_dates_inv_org.sort()
# List_dates_inv.sort()

# for i in range(len(List_dates_unique)):
#     List_dates_unique[i]=List_dates_unique[i].replace("T21:00:00Z","")
#     List_dates_unique[i]=List_dates_unique[i].replace("-","")

# for i in range(len(List_dates_inv)):
#     List_dates_inv[i]=List_dates_inv[i].replace("T21:00:00Z","")
    



# ############### Putting all the available observation dates in a list

# fl_obs = glob.glob(data_dir_obs+"/obs*")

# len_fl_obs = len(fl_obs)

# for i in range(len_fl_obs):
#     fl_obs[i] = fl_obs[i].replace(data_dir_obs, "")
#     fl_obs[i] = fl_obs[i].replace(".npy", "")
#     fl_obs[i] = fl_obs[i].replace("obs", "")
    
#     fl_obs[i] = fl_obs[i][0:8]
#     #print(fl_obs[i])

# fl_obs=list(set(fl_obs))

# fl_obs.sort()

#####################################

def split_dataset(file_list,N_parts):
    """
    randomly separate a list of files in N_parts distinct parts
    
    Inputs :
        file_list : a list of filenames
        
        N_parts : int, the number of parts to split on
    
    Returns :
         list of N_parts lists of files
    
    """
    
    inds=[ i * len(file_list)//N_parts for i in range(N_parts)] + [len(file_list)]

    to_split = file_list.copy()
    random.shuffle(to_split)
    
    return [to_split[inds[i]:inds[i+1]] for i in range(N_parts)]


def match_ensemble_to_samples(indices, df, data_dir, mode = 'Ens2Samples', List_dates_inv_org=None):
    """
    fetch corresponding filenames from one dataset preparation to another
    (i.e from _samples to ensembles or reverse, depending on the mode)
    --> prepare a set of indices to get the corresponding data in the same order
    as provided by indices
    """
    file_names = []
    
    if mode=='Ens2Samples' :
        ## give the names of the samples used in a given list of ensembles
        ## we assume the structure of indices is a list of 
        ##[(int--> date, int -->lead_time)]
        
        for ind in indices : #each ind corresponds to an ensemble file
            ind_i = (int(ind[0]), int(ind[1])) 
            names = df[(df['Date']==List_dates_inv_org[ind_i[0]]) & (df['LeadTime']==(ind[1]+1)*3-1)]['Name'].to_list()
            file_names.append([data_dir + n + '.npy' for n in names])

            
    
    if mode=='Sample2Ens' :
        ## give the names of each Ensembles corresponding to each sample if the list
        ## we assume the structure of indices is a list of (int --> '_sampleInt.npy')
        ##[(int--> date, int -->lead_time)]
        for ind in indices :
            name = '_sample'+str(ind)
            
            data =  df[df['Name']==name]
            
            fname = 'Fsemble_'+str(data['DateIndex'])+'_'+str(data['LeadTime'])+'.npy'
            
            file_names.append(fname)
        
    return file_names


def normalize(BigMat, scale, Mean, Max):
    
    """
    
    Normalize samples with specific Mean and max + rescaling
    
    Inputs :
        
        BigMat : ndarray, samples to rescale
        
        scale : float, scale to set maximum amplitude of samples
        
        Mean, Max : ndarrays, must be broadcastable to BigMat
        
    Returns :
        
        res : ndarray, same dimensions as BigMat
    
    """
    
    res= scale*(BigMat-Mean)/(Max)

    return  res


def build_datasets(data_dir, program, option='fake', indexList=None,
                   data_option='ens', dh = None, N_runs = None, List_dates_inv=None, data_dir_real=None,
                   List_dates_inv_org=None, inv_step = None, conditioning_members=None, N_ensemble=None):
    """
    
    Build file lists to get samples, as specified in the program dictionary
    
    Inputs :
        
        data_dir : str, the directory to get the data from
        
        program : dict,the datasets to be constructed
                dict { dataset_id : (N_parts, n_samples)}
                
        option : str, optional  -> whether samples originate from a 'real' dataset
        
        indexList : list, optional -> the indexes of samples to be chosen
        
        data_option : str, optional, for fake data only -> whether data is grouped
        into ensembles or dispatched into samples
    
    Returns :
        
        res, dictionary of the shape {dataset_id : file_list}
        
        !!! WARNING !!! : the shape of file_list depends on the number of parts
        specified in the "program" items. Can be nested.
    
    """
    df = pd.read_csv(data_dir_real + 'Large_lt_test_labels.csv')    
    res = {}
    
    for key, value in program.items():
        
         print(key, value)
        
         if value[0]==1:
            

            N_samples = value[1] 
            
            if indexList is None :
            
                indexList = list(range(N_samples))[:N_samples]

            indices = [(i//N_runs, i%N_runs) for i in indexList]
            print(indices, value[1])
            if option == 'observation':
                
                fileList = [data_dir + 'obs_'+str(ind[0])+'_'+str(ind[1])+'.npy'
                            
                            for ind in indices ]
            
            if option=='real' : ### we are going to gather individual samples
                
                fileList = match_ensemble_to_samples(indices, df, data_dir, List_dates_inv_org=List_dates_inv_org)
            
            elif option=='fake' and data_option=='ens':
                
                #fileList = [data_dir + 'Rsemble_'+List_dates_inv[ind[0]]+'_'+str((ind[1]+1)*3)+'.npy' for ind in indices ]
                #fileList = [data_dir + 'invertFsemble_'+List_dates_inv[ind[0]]+'_'+str((ind[1]+1)*3) + '_' + str(inv_step) +'.npy' for ind in indices ]

                #fileList = [data_dir + 'invertFsemble_'+List_dates_inv[ind[0]]+'_'+str((ind[1]+1)*3)+'_1000'+'.npy' for ind in indices ]
                #fileList = [data_dir + 'genFsemble_'+List_dates_inv[ind[0]]+'_'+str((ind[1]+1)*3)+'_' + str(inv_step) + '_'
                #            + str(conditioning_members) + '_' + str(N_ensemble) + '.npy' for ind in indices ]
                fileList = [data_dir + 'genFsemble_'+List_dates_inv[ind[0]]+'_'+str((ind[1]+1)*3)+'_' + str(inv_step) + '.npy' for ind in indices ]            
            
            elif option=='fake' and data_option=='sample' :
                
                fileList = match_ensemble_to_samples(indices, data_dir, df)
            
            elif option=='fake_latent' :
                
                fileList = [data_dir + 'w_'+str(float(ind[0]))+'_'+str(float(ind[1]))+'.npy'
                            
                            for ind in indices ]
            
            res[key] = fileList
        
         if value[0]==2:
            
            N_samples = value[1] 
            
            if indexList is None :
            
                indexList =  list(range(N_samples))[:N_samples]
                
            indices = [(i//8, i%8) for i in indexList]
            
            if option=='real' : ### we are going to gather individual samples
                
                fileList = match_ensemble_to_samples(indices, data_dir, df)
            
            elif option=='fake' and data_option=='ens' :
                
                fileList = [data_dir + 'Fsemble_'+str(float(ind[0]))+'_'+str(float(ind[1]))+'.npy'
                            
                            for ind in indices ]
            
            elif option=='fake' and data_option=='sample' :
                
                fileList = match_ensemble_to_samples(indices, data_dir, df)
            
            elif option=='fake_latent' :
                
                fileList = [data_dir + 'w_'+str(float(ind[0]))+'_'+str(float(ind[1]))+'.npy'
                            
                            for ind in indices ]
            
            res[key] = split_dataset(fileList,2)
    print(indexList)
    print('built datasets', option)
    print('Here are all the files that are read', fileList)
    print(res[key][0])
    #print(res[key])
    
    return res, indexList


def load_batch(file_list, number,\
               var_indices_real = None, var_indices_fake = None,
               crop_indices = None,
               option = 'real', subsample=16, deterministic=True, N_runs=None, dh=None, fl_obs=None,
               List_dates_unique=None, data_dir_obs=None):
     
    """
    gather a fixed number of random samples present in file_list into a single big matrix

    
    Inputs :
        
        file_list : list of files to be sampled from
        
        number : int, the number of samples to draw
        
        var_indices(_real/_fake) : iterable of ints, coordinates of variables in a given sample to select
        
        crop_indices : iterable of ints, coordinates of data to be taken (only in 'real' mode)
                
        Shape : tuple, the target shape of every sample
        
        option : str, different treatment if the data is GAN generated or PEARO
        
        subsample : int, whether to sample a number of members from each
        loaded batch. default to 16 --> no subsampling
        
        deterministic : bool, whether the subsampling is deterministic (first subsample members)
        or random
    
    Returns :
        
        Mat : numpy array, shape  number x C x Shape[1] x Shape[2] matrix
        
    """
    
    
    if option=='fake':
        # in this case samples can either be in isolated files or grouped in batches

        assert var_indices_fake is not None # sanity check
        
        if len(var_indices_fake) ==1 :                    
            ind_var = var_indices_fake[0]
            
        test_struct = file_list[0]

        if type(test_struct)==list :
            fn = test_struct[0]
        elif type(test_struct)==str:
            fn = test_struct
        Shape = np.load(fn).shape


        ## case : isolated files (no batching)
        ## one creates another dimension to group by ensemble
        
        if len(Shape)==3:
            
            Mat = np.zeros((number,subsample,len(var_indices_fake), Shape[1], Shape[2]), dtype=np.float32)
            
            if subsample < 16 :
                if deterministic :
                    indices = list(range(subsample))
                else:
                    indices = random.sample(list(range(16)), subsample)
            else :
                indices = list(range(16))
            
            for i in range(number) :
                for j in indices: # iterating over ensemble indices
                    if len(var_indices_fake)==1 :
                        Mat[i,j] = np.load(file_list[i][j])[ind_var:ind_var+1,:,:].astype(np.float32)
                    else :    
                        Mat[i,j] = np.load(file_list[i][j])[var_indices_fake,:,:].astype(np.float32)
        
        ## case : batching -> select the right number of files to get enough samples
        ## batching is always assumed to gather a single ensemble
        ## in this case the 'number' parameter is just the number of files to load
        
        elif len(Shape)==4 and type(test_struct)==str:
            
            #select multiple files and fill the number
        
            Mat = np.zeros((number, subsample, len(var_indices_fake), Shape[2], Shape[3]), \
                                                     dtype=np.float32)
            
            for i in range(number) :

                if deterministic :
                    indices = list(range(subsample))
                else:
                    indices = random.sample(list(range(16)), subsample)   

                    
                if len(var_indices_fake)==1 :
                    
                    dat =  np.load(file_list[i]).astype(np.float32)[indices]
                    Mat[i] = dat[:,ind_var:ind_var+1,:,:]
                
                else :
                    dat = np.load(file_list[i]).astype(np.float32)[indices]
                    Mat[i] = dat[:,var_indices_fake,:,:] #\
                #np.load(file_list[i]).astype(np.float32)[indices,var_indices_fake,:,:]
                
        elif len(Shape)==4 and type(test_struct)==list:
            print(Shape, len(file_list), len(test_struct))
            #select multiple files and fill the number
        
            Mat = np.zeros((number, subsample, len(var_indices_fake), Shape[2], Shape[3]), \
                                                     dtype=np.float32)
            print(Mat.shape)
            
            for i in range(number) :
                
                if deterministic :
                    indices = list(range(subsample))
                else:
                    indices = random.sample(list(range(16)), subsample)
                    
                
                if len(var_indices_fake)==1 :
                    
                    dat =  np.load(file_list[0][i]).astype(np.float32)[indices]
                    Mat[i] = dat[:,ind_var:ind_var+1,:,:]
                
                else :
                    dat = np.load(file_list[0][i]).astype(np.float32)[indices]
                    Mat[i] = dat[:,var_indices_fake,:,:] #\
            #np.load(file_list[i]).astype(np.float32)[indices,var_indices_fake,:,:]

    elif option=='real':
        

        # in this case samples are stored once per file
        
        Shape=(len(var_indices_real),
               crop_indices[1]-crop_indices[0],
               crop_indices[3]-crop_indices[2])
        

        Mat = np.zeros((number,min(subsample,16), Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        

        for i in range(number):
            if deterministic :
                indices = list(range(min(16,subsample)))
            else:
                indices = random.sample(list(range(16)), subsample)

            for j in indices:
                if len(var_indices_real)==1 :
                
                    ind_var = var_indices_real[0]
                
                    Mat[i,j] = np.load(file_list[i][j])[ind_var:ind_var+1,
                                           crop_indices[0]:crop_indices[1],
                                           crop_indices[2]:crop_indices[3]].astype(np.float32)
                else :

                    #Mat[i,j] = np.load(file_list[i][j])[var_indices_real, ### ATTENTION commented because this does not work if the original database is not AROME full
                    #                       crop_indices[0]:crop_indices[1],
                    #                       crop_indices[2]:crop_indices[3]].astype(np.float32)
                    
                    Mat[i,j] = np.load(file_list[i][j])[var_indices_real].astype(np.float32)
    elif option=='obs':
        
        # in this case samples are stored once per file

        Shape=(len(var_indices_real),
               crop_indices[1]-crop_indices[0],
               crop_indices[3]-crop_indices[2])
        
        Mat = np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        
        
        k = 0
        
        for k in range(number):
            
            D_index = k//int(N_runs)
            
            LT_index = k%int(N_runs)
            print(N_runs, D_index, LT_index, dh)
            
            hour_LT = (LT_index)*dh
            
            
            if hour_LT > 23 :
                
                hour = (LT_index)*dh - 24
            else : 
                
                hour = (LT_index)*dh
                    
            if hour_LT > 23:
                obs = np.load(data_dir_obs+"obs" + fl_obs[fl_obs.index(List_dates_unique[D_index])+2] + "_" +str(hour) + ".npy") # the +1 is needed since the simulation starts at 21:00 the day-1 first observations are available at 00:00 the next day
                
            else:
                obs = np.load(data_dir_obs+"obs" + fl_obs[fl_obs.index(List_dates_unique[D_index])+1] + "_" +str(hour) + ".npy")
            
            
            obs = oc.obs_clean(obs, crop_indices)
            
            Mat[k] = obs
            print("OBS TO GET", fl_obs[fl_obs.index(List_dates_unique[D_index])+2], fl_obs[fl_obs.index(List_dates_unique[D_index])+1])
    


                

    return Mat



def eval_distance_metrics(data):
    
    """
    
    this function should test distance metrics for datasets=[(filelist1, filelist2), ...]
    in order to avoid memory overhead, datasets are created and destroyed dynamically
    
    Inputs :
        
       data : tuple of 
       
           metric : str, the name of a metric in the metrics4arome namespace
           
           dataset : 
               dict of file lists (option='from_names')
               
               Identifies the file names to extract data from
               
               Keys of the dataset are either : 'real', 'fake' when comparing 
                               sets of real and generated data
                               
                                                'real0', 'real1' when comparing
                               different sets of real data
                               
                                                 'fake0', 'fake1' when comparing
                               different sets of real data
           
           n_samples_0, n_samples_1 : int,int , the number of samples to draw
                                      to compute the metric.
                                      
                                      Note : most metrics require equal numbers
                                      
                                      by default, 0 : real, 1 : fake
            
           VI, CI : iterables of indices to select (VI) variables/channels in samples
                                                   (CI) crop indices in maps
               
           index : int, identifier to the data passed 
                   (useful only if used in multiprocessing)

       
       option : str, to choose if generated data is loaded from several files (from_names)
               or from one big Matrix (from_matrix)
                   
    Returns :
        
        results : np.array containing the calculation of the metric
        
    """
    
    #print(data)
    metrics_list, dataset, n_samples_0, n_samples_1, VI, VI_f, CI, index, subsample, parameters, N_runs, dh, debiasing, conditioning_members, data_dir_real, data_dir_obs,List_dates_inv, sList_dates_inv_org, List_dates_unique,fl_obs= data
    print('Subsample', subsample)
    ## loading and normalizing data
    
    print('loading constants')
    Means = np.load(data_dir_real + 'Mean_4_var.npy')[VI].reshape(1,len(VI),1,1)
    Maxs = np.load(data_dir_real + 'MaxNew_4_var.npy')[VI].reshape(1,len(VI),1,1)

    if type(subsample)==tuple:
        real_sub, fake_sub = subsample
    else :
        real_sub, fake_sub = subsample, subsample
    
    if list(dataset.keys())==['real','fake']:
    
        print('index',index)
            
        print('load fake')  
        fake_data = load_batch(dataset['fake'], n_samples_1, 
                               var_indices_fake = VI_f, option='fake',
                               subsample = fake_sub, fl_obs=fl_obs,
               List_dates_unique=List_dates_unique, data_dir_obs=data_dir_obs)
        fake_data = oc.denorm(fake_data, Maxs, Means, 0.95)
        print('load real')
        real_data = load_batch(dataset['real'], n_samples_0, 
                               var_indices_real = VI, var_indices_fake = VI_f, 
                               crop_indices = CI, subsample = real_sub, fl_obs=fl_obs,
               List_dates_unique=List_dates_unique, data_dir_obs=data_dir_obs)
        
        #real_data = normalize(real_data, 0.95, Means, Maxs)
        print(real_data.shape, fake_data.shape)
        
    elif list(dataset.keys())==['obs','fake','real']: ######### HERE the magic happens

        print('index',index)
            
        print('load fake')  
        fake_data = load_batch(dataset['fake'], n_samples_1, 
                               var_indices_fake = VI_f, option='fake',
                               subsample = fake_sub, fl_obs=fl_obs,
               List_dates_unique=List_dates_unique, data_dir_obs=data_dir_obs) # load_batch = load_dataset
        fake_data = fake_data.astype(np.float32)
        print('load real')
        real_ens = load_batch(dataset['real'], n_samples_0, 
                               var_indices_real = VI, var_indices_fake = VI_f, 
                               crop_indices = CI, subsample = real_sub, fl_obs=fl_obs,
               List_dates_unique=List_dates_unique, data_dir_obs=data_dir_obs)
        
        print(real_ens.shape, real_ens.max())
        real_ens = real_ens.astype(np.float32)
        print('load obs')
        fake_data = oc.denorm(fake_data, Maxs, Means, 0.95)
        fake_data = fake_data.astype(np.float32)
        real_data = load_batch(dataset['obs'], n_samples_0, 
                               var_indices_real = VI, var_indices_fake = VI_f, 
                               crop_indices = CI, option = 'obs', subsample = real_sub, N_runs=N_runs, dh=dh, fl_obs=fl_obs,
               List_dates_unique=List_dates_unique, data_dir_obs=data_dir_obs)
        
        real_data = real_data.astype(np.float32)
        
        std_by_station = np.nanstd(real_data, axis=(0), ddof = 1)
        std = np.nanmean(std_by_station, axis=(1,2))
        mean = np.nanmean(real_data, axis=(0,2,3))
        
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        q_obs_by_station = np.nanquantile(real_data, quantiles, axis = (0))
        q_obs = np.nanmean(q_obs_by_station, axis=(2,3) )
        print(q_obs_by_station.shape)
        print(mean, std, q_obs)
        #print(stoppp)
        #real_data = normalize(real_data, 0.95, Means, Maxs)
        
        print(real_data.shape, fake_data.shape)
        
    elif list(dataset.keys())==['real0', 'real1']:

        print('index', index)
    
        real_data0 = load_batch(dataset['real0'],n_samples_0, 
                                var_indices_real = VI, crop_indices = CI,
                                subsample = real_sub)
        real_data1 = load_batch(dataset['real1'], n_samples_1, 
                                var_indices_real = VI, crop_indices = CI,
                                subsample = fake_sub)
        
        real_data = normalize(real_data0, 0.95, Means, Maxs)
        fake_data = normalize(real_data1, 0.95, Means, Maxs)  # not stricly "fake" but same
    
    elif list(dataset.keys())==['fake0', 'fake1']:
    
        print('index', index)
    
        real_data = load_batch(dataset['fake0'], n_samples_0, 
                               var_indices_fake = VI_f, crop_indices = CI,
                               subsample = real_sub) # not strictly 'real' but same
        
        fake_data = load_batch(dataset['fake1'], n_samples_1, 
                               var_indices_fake = VI_f, crop_indices = CI,
                               subsample = fake_sub)
    
        
    else :
        raise ValueError("Dataset keys must be either 'real'/'fake', \
                         'real0'/'real1', 'fake0'/'fake1', not {}"
                         .format(list(dataset.keys())))
        
    ## the interesting part : computing each metric of metrics_list
    
    results = {}
    
    for metric in metrics_list :
    
        print(metric)
        if list(dataset.keys())==['obs','fake','real']:
            if metric == 'brier_score' or metric == 'rel_diagram' : 
                Metric = getattr(metrics, metric)
        
                results[metric] = Metric(real_data, fake_data, real_ens, parameters = parameters, debiasing = debiasing, conditioning_members=conditioning_members, select = False)
                
            else : 
               Metric = getattr(metrics, metric)
        
               results[metric] = Metric(real_data, fake_data, real_ens, debiasing = debiasing, conditioning_members=conditioning_members, select = False) 
        else:
            Metric = getattr(metrics, metric)
            print(Metric)
            results[metric] = Metric(real_data, fake_data, select = False)
                
    return results, index

def global_dataset_eval(data):
    
    """
    
    evaluation of metric on the DataSet (treated as a single numpy matrix)
    
    Inputs :
    
        data : iterable (tuple)of str, dict, int
            
            metric : str, the name of a metric in the metrics4arome namespace
            
            dataset :
                file list /str containing the ids of the files to get samples
                
            
            n_samples_0, n_samples_1 : int,int , the number of samples to draw
                                          to compute the metric.
                                          
                                          Note : most metrics require equal numbers
                                          
                                          by default, 0 : real, 1 : fake
                
            VI, CI : iterables of indices to select (VI) variables/channels in samples
                                                       (CI) crop indices in maps
                   
            index : int, identifier to the data passed 
                       (useful only if used in multiprocessing)
                       
            subsample :  int to choose random members from the ensemble. 
            To avoid raising an error, if tuple is provided, 
            only first element is considered

    
    Returns :
        
        results : dictionary contining the metrics list evaluation
        
        index : the index input (to keep track on parallel execution)
        
    """
    
    metrics_list, dataset, n_samples, VI, VI_f, CI, index, data_option, subsample = data
    print('Subsample', subsample)
    print('index', index)
    
    if type(subsample)==tuple: # 
        real_sub = subsample[0]
    else :
        real_sub = subsample
        
    if data_option=='fake' :
        print('loading fake data')
        assert(type(dataset[0])==list)
    
        rdata = load_batch(dataset, n_samples,
                           var_indices_fake = VI_f,
                           crop_indices = CI, option = data_option, subsample = real_sub)

    elif data_option=='real' :
       print('loading real data')
       assert(type(dataset[0])==list)
       
       rdata = load_batch(dataset[0], n_samples,
                          var_indices_real = VI,
                          crop_indices = CI, option=data_option, subsample = real_sub)
       print('real data loaded, shape ', rdata.shape)
    
    if data_option=='real':
        print('normalizing')
        Means = np.load(data_dir_real+'mean_with_orog.npy')[VI].reshape(1,len(VI),1,1)
        Maxs = np.load(data_dir_real+'max_with_orog.npy')[VI].reshape(1,len(VI),1,1)
    
        rdata = normalize(rdata, 0.95, Means, Maxs)
        
    results = {}
    
    for metric in metrics_list :
    
        print(metric)
        
        Metric = getattr(metrics, metric)
        
        results[metric] = Metric(rdata, select = False)
    return results, index

