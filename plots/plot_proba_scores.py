
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pickle
from torchvision import transforms
import copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import matplotlib.gridspec as gridspec


def plot_mean_var_1l(Ens_proj_var, ref, name):
    
    fig, axs = plt.subplots(2, 3, figsize=(6, 1.4))
    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(wspace=0.175, hspace=0.175) # set the spacing between axes. 
    
    #plt.rcParams["figure.figsize"] = [50.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    
    axs = plt.subplot(gs1[0])
    c = axs.pcolor(Ens_proj_var[2,:,:], cmap="coolwarm", vmin=-0.1, vmax=0.1)
    axs.set_title('t2m')
    axs.axis('off')
    cb = fig.colorbar(c,ax=axs)     
    #cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')   
    cb.ax.tick_params(labelsize=8)    
    cb.ax.yaxis.get_offset_text().set(size=8)                  
    cb.update_ticks()
    

    
    axs = plt.subplot(gs1[1])
    c = axs.pcolor(Ens_proj_var[0,:,:], cmap="coolwarm", vmin=-0.1, vmax=0.1)
    axs.axis('off')
    axs.set_title('ff')
    cb = fig.colorbar(c,ax=axs)     
    #cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right') 
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)                         
    cb.update_ticks()
    

    
    
    axs = plt.subplot(gs1[2])
    c = axs.pcolor(Ens_proj_var[1,:,:], cmap="coolwarm", vmin=-0.1, vmax=0.1)
    axs.axis('off')
    axs.set_title('dd')
    cb = fig.colorbar(c,ax=axs)     
    #cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.get_offset_text().set(size=8)                          
    cb.update_ticks()
    

    
    
    plt.savefig(name + ".jpg",  dpi=600, transparent=False, bbox_inches='tight')


def plots_ens(tests_list, Path_to_q, n_q, N_e, n_c, size_x):
    
    Means = np.load('/scratch/mrmn/moldovang/IS_1_1.0_0_0_0_0_0_256_done/mean_with_orog.npy')[1:4].reshape(3,1,1)
    Maxs = np.load('/scratch/mrmn/moldovang/IS_1_1.0_0_0_0_0_0_256_done/max_with_orog.npy')[1:4].reshape(3,1,1)
    
    print(Means, Maxs)
    Stds = (1.0/0.95) * Maxs
    data_dir = '/scratch/mrmn/moldovang/IS_1_1.0_0_0_0_0_0_256_done/'
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize(Means, Stds),
        ]
    )
    
    data = np.load(data_dir + '_sample1' + '.npy')[0:5,78:206,55:183].astype(np.float32)
    data = data.transpose((1,2,0))
    img = transform(data)
    
    sea_indices = []
    mountain_indices = []
    plain_indices = []
    for i in range(128):
        for j in range(128):
            
            if img[4,i,j] == 0:
                
                sea_indices.append([i,j])
            elif img[4,i,j] >  1000.:
                mountain_indices.append([i,j])
            
            else :
                plain_indices.append([i,j])
                
                
    sea_indices = np.array(sea_indices)
    mountain_indices = np.array(mountain_indices)
    plain_indices = np.array(plain_indices)

    
    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 25,
        }
    len_tests = len(tests_list)
    name_res = '/log/distance_metrics_distance_metrics_592.p'

    name_res_rd = '/log/rel_diagram_distance_metrics_592.p'
    N_bins_max = 121
    
    Brier_scores = np.zeros((len_tests, N_e, 6, n_c, size_x, size_x), dtype = 'float32')
    Brier_scores_LT = np.zeros((len_tests, 74, 8, 6, n_c, size_x, size_x), dtype = 'float32')
    
    s_p_scores = np.zeros((len_tests, N_e, 2, n_c, size_x, size_x), dtype = 'float32')
    s_p_scores_LT = np.zeros((len_tests, 74, 8, 2, n_c, size_x, size_x), dtype = 'float32')
    
    crps_scores = np.zeros((len_tests, N_e, n_c, size_x, size_x), dtype = ('float32'))
    crps_scores_LT = np.zeros((len_tests, 74, 8, n_c, size_x, size_x), dtype = 'float32')
    
    mean_bias = np.zeros((len_tests, N_e, n_c, size_x, size_x), dtype = ('float32'))
    mean_bias_LT = np.zeros((len_tests, 74, 8, n_c, size_x, size_x), dtype = 'float32')
    
    bins = np.linspace(0, 1, num=11)
    freq_obs = np.zeros((10))
    Hit_rate = np.zeros((17))
    false_alarm = np.zeros((17))
    
    A_ROC = np.zeros((len_tests))
    A_ROC_skill = np.zeros((len_tests))
    
    Hit_rate[0]=0
    Hit_rate[16]=1
    false_alarm[0]=0
    false_alarm[16]=1
    
    #rel_diag_scores = np.zeros((len_tests,N_e, 6, 3, 2, 10))
    rel_diag_scores = np.zeros((len_tests,N_e, 6, 2, n_c, size_x, size_x))
    rank_histo = np.zeros((len_tests, N_e, n_c, N_bins_max))
    
    

    for i in range(len_tests):
        
        res = pickle.load(open(Path_to_q + tests_list[i] + name_res, 'rb'))
        brier = res['brier_score']
        
        res = pickle.load(open(Path_to_q + tests_list[i] + name_res, 'rb'))
        crps = res['ensemble_crps']
        
        res = pickle.load(open(Path_to_q + tests_list[i] + name_res, 'rb'))
        res_rd = pickle.load(open(Path_to_q + tests_list[i] + name_res_rd, 'rb'))
        s_p = res['skill_spread']
        rd = res_rd['rel_diagram']
        r_h = res['rank_histogram']
        m_bias = res['bias_ensemble']
        #print(m_bias)
        Brier_scores[i] = brier
        crps_scores[i] = crps
        s_p_scores[i] = s_p
        rel_diag_scores[i] = rd
        rank_histo[i] = r_h
        mean_bias[i] = m_bias
        
    """
    Transformation to identify Lead Times more easily

    """
    D_i = 0
    LT_i = 0
    for i in range(592):        
        
        Brier_scores_LT[:,D_i, LT_i] = Brier_scores [:, i]
        crps_scores_LT[:, D_i, LT_i] = crps_scores[:,i]
        s_p_scores_LT[:, D_i, LT_i] = s_p_scores[:,i]
        mean_bias_LT[:,D_i, LT_i] = mean_bias[:,i]
        LT_i =LT_i + 1
        
        if LT_i == 8 : 
            
            D_i = D_i +1
            LT_i = 0
        
        
    #############################################################################################
    
    
    color_p = ['black', 'royalblue', 'darkgreen', 'darkorange', 'red', 'cyan', 'gold', 'pink', 'tan', 'slategray', 'purple', 'palegreen']

    case_name = [['ff=3 (m/s)', 'ff=4 (m/s)', 'ff=5 (m/s)', 'ff=6 (m/s)', 'ff=7 (m/s)', 'ff=8 (m/s)'],
                 ['', '', '', '', '', ''], 
                 ['t2m=278.15 (K)', 't2m=281.15 (K)', 't2m=283.15 (K)', 't2m=285.15 (K)', 't2m=287.15 (K)', 't2m=289.15 (K)']]

    case_name = [['ff=5 (m/s)', 'ff=7.5 (m/s)', 'ff=10 (m/s)', 'ff=12.5 (m/s)', 'ff=15 (m/s)', 'ff=17.5 (m/s)'],
                 ['', '', '', '', '', ''], 
                 ['t2m=278.15 (K)', 't2m=283.15 (K)', 't2m=288.15 (K)', 't2m=293.15 (K)', 't2m=298.15 (K)', 't2m=303.15 (K)']]

    remove_comm = 'rm -r -f ' + 'brier crps ROC rel_diag mean_bias skill_spread rank_histo'
    os.system(remove_comm)
    os.mkdir('brier') 
    os.mkdir('crps') 
    os.mkdir('ROC') 
    os.mkdir('rel_diag')
    os.mkdir('mean_bias')
    os.mkdir('skill_spread')
    os.mkdir('rank_histo')
    
    
    var_names = ['ff', 'dd', 't2m']
    var_names_m = ['ff (m/s)', 'dd (°)', 't2m (K)'  ]
    domain = ['sea', 'plain', 'mountain']
    cases_clean = ['AROME', 'Style Perturbation', 'PCA Resampling', 'Extrapolation']
    cases_clean = ['AROME', '1000_0.1', '400_0.1', '400_0.1_NSEOFF', '400_0.25']
    cases_clean = ['AROME', '1000_0.1', '400_0.25', '400_0.25_NSEOFF', '400_0.25_NSEON']
    cases_clean = ['AROME', '200_1', '200_1*', '400_1', '400_1*', '400_2', '400_3', '400_3*', '400_4', '400_4*', '1000']
    cases_clean = ['AROME', '1', '5', '10', '20']
    cases_clean = ['AROME', '2_3', '2_4', '2_4_biased', '1_3']
    cases_clean = ['AROME', '2_3', '2_4','1_4']
    cases_clean = ['AROME', '1','20', '40']
    cases_clean = ['AROME', '1', '1_NO', '5', '10', '20', '40']
    cases_clean = ['AROME', 'B', 'B_2_3']
    cases_clean = ['AROME', '0.25', '0.1']
    #cases_clean = ['AROME', 'from_z', 's_w_p_f']
    #cases_clean = ['AROME', '1000', '200', '300_N', '300_Z', '300_R']
    echeance = ['+3H', '+6H', '+9H', '+12H', '+15H', '+18H', '+21H', '+24H']
################################################ MEAN BIAS    
    for i in range(n_c):
        

        fig,axs = plt.subplots(figsize = (9,7))
        
        for k in range(len_tests):
                
            #plt.ylim([-0.3, 0.3])
            print(mean_bias_LT[k,:,:,i].shape, )
            #plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i, indices[:,0], indices[:,1]]**2., axis =(0,1)))/np.nanmean(s_p_scores_LT[k,:,:,1,i, indices[:,0], indices[:,1]], axis =(0,1)), label = tests_list[k], color= color_p[k] )
            #plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i, indices[:,0], indices[:,1]]**2., axis =(0,1))), label = tests_list[k], color= color_p[k] )
            plt.plot(np.nanmean(mean_bias_LT[k,:,:,i], axis = (0,2,3)), label = cases_clean[k], color= color_p[k] )


            #plt.plot(np.nanmean(s_p_scores_LT[k,:,:,1,i, indices[:,0], indices[:,1]], axis =(0,1)), linestyle='dashed', color= color_p[k] )
            axs.set_xticks(range(len(echeance)))
            axs.set_xticklabels(echeance)
            plt.xticks( fontsize ='18')
            axs.tick_params(direction='in', length=12, width=2)

            plt.yticks(fontsize ='18')
            #plt.title(var_names[i] ,fontdict = font)
            #plt.text(0.6, 0.9, var_names[i] + ' ' + domain[ii],
              #        fontdict = font, transform=axs.transAxes)
            plt.ylabel(var_names_m[i], fontsize= '18')
            plt.legend(fontsize = 14,frameon = False, ncol=2)
            plt.savefig('/scratch/mrmn/moldovang/score_ensemble/plots/mean_bias/mean_bias'+str(i)+'.png')
        




    
    
    
################################################################# PLOT RANK HISTOGRAM    
    N_bins= [17, 113, 121, 121]
    N_bins= [17,113,113, 113, 113]
    N_bins= [17,17, 17, 17, 17, 17, 17, 17, 17, 17, 17]
    N_bins= [17,113, 113, 113, 113, 113, 113, 113, 113, 113, 113]
    N_bins= [17,113,113, 113]
    N_bins= [17,113,113, 113, 113, 113, 113]
    #N_bins= [17,120,120]
    N_bins= [17,121,121]
    N_bins= [17,113,113]
    for j in range(n_c):
        for k in range(len_tests):
            fig,axs = plt.subplots(figsize = (9,7))
            ind = np.arange(N_bins[k])
            print(rank_histo[k,:,j,0:N_bins[k]].sum(axis=0).shape)
            plt.bar(ind, rank_histo[k,:,j,0:N_bins[k]].sum(axis=0))
            plt.title(cases_clean[k] + ' ' + var_names[j],fontdict = font)
            #plt.xticks( fontsize ='18')
            plt.tick_params(bottom = False, labelbottom = False)
            plt.xlabel('Bins', fontsize= '18')
            plt.ylabel('Number of Observations', fontsize= '18')
            axs.tick_params(length=12, width=2)
            plt.yticks(fontsize ='18')

            plt.savefig('/scratch/mrmn/moldovang/score_ensemble/plots/rank_histo/rank_histo'+str(j)+'_'+str(k)+'.png')

        #plt.xticks( fontsize ='18')
        #plt.xlabel('forecast probability', fontsize= '18')
        #plt.ylabel('observation frequency', fontsize= '18')
        #axs.tick_params(direction='in', length=12, width=2)
        #plt.yticks(fontsize ='18')
        #plt.text(0.3, 0.9, case_name[j][i],
        #         fontdict = font, transform=axs.transAxes)
        #plt.legend(fontsize = 14,frameon = False, ncol=2)
        





    
#####################################################"PLOT REL DIAGRAM

    for i in range(6):
        
        for j in range(n_c):
            fig,axs = plt.subplots(figsize = (9,7))
            for k in range(len_tests):
                O_tr = rel_diag_scores[k,:-2,i,1,j]
                X_prob = rel_diag_scores[k,:-2,i,0,j]
                #print(O_tr.shape, X_prob.shape,)
                
                for z in range(bins.shape[0]-1):
                    
                    obs = copy.deepcopy(O_tr[np.where((X_prob >= bins[z]) & (X_prob < bins[z+1]), True, False)])
                    obs = obs[~np.isnan(obs)]
                    print(obs.shape, j)
                    freq_obs[z] = obs.sum()/obs.shape[0]
                plt.plot(bins[:-1]+0.05, freq_obs, label = cases_clean[k], color = color_p[k])
            
            plt.plot(bins[:-1]+0.05, bins[:-1]+0.05, label = 'perfect', color = 'black')
            #plt.ylim([-0.15, 0.15])
            plt.xticks( fontsize ='18')
            plt.xlabel('forecast probability', fontsize= '18')
            plt.ylabel('observation frequency', fontsize= '18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[j][i],fontdict = font)
            #plt.text(0.3, 0.9, case_name[j][i],
            #         fontdict = font, transform=axs.transAxes)
            plt.legend(fontsize = 14,frameon = False, ncol=2)
            plt.savefig('/scratch/mrmn/moldovang/score_ensemble/plots/rel_diag/rel_diag'+str(i)+'_'+str(j)+'.png')
            







#############################################"" PLOT CRPS####################################


    for i in range(n_c):
        
        #for ii in range(3):
        #    if ii == 0:
        #        indices = sea_indices
        #    elif ii == 1:
        #        indices = plain_indices
        #    elif ii == 2:
        #        indices = mountain_indices
            fig,axs = plt.subplots(figsize = (9,7))
            
            for k in range(len_tests):
                    
                #plt.ylim([-0.3, 0.3])
                #plt.plot(np.nanmean(crps_scores_LT[k,:,:,i,indices[:,0], indices[:,1]], axis=(0,1)), label=tests_list[k], color=color_p[k] )
                print(crps_scores_LT[k,:,:,i].shape)
                plt.plot(np.nanmean(crps_scores_LT[k,:,:,i], axis=(0,2,3)), label=cases_clean[k], color=color_p[k] )
            
            plt.xticks( fontsize ='18')
            axs.set_xticks(range(len(echeance)))
            axs.set_xticklabels(echeance)
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            #plt.title(var_names[i] + ' ' + domain[ii],fontdict = font)
            plt.title(var_names[i] ,fontdict = font)                
            #plt.text(0.6, 0.9, var_names[i] + ' ' + domain[ii],
            #        fontdict = font, transform=axs.transAxes)
            plt.legend(fontsize = 14,frameon = False, ncol=2)
            plt.savefig('/scratch/mrmn/moldovang/score_ensemble/plots/crps/crps'+str(i) +'.png')
            #print(np.nanmean(crps_scores_LT[4,:,:,:], axis=(0,1)).shape)
            #diff = crps_scores_LT[4,:,:,:] - crps_scores_LT[0,:,:,:]
            
            #diff_mean = np.nanmean(diff, axis=(0,1))
            
            #diff_mean[diff_mean>0]=1
            #diff_mean[diff_mean<0]=-1
            
            
            #plot_mean_var_1l(diff_mean, diff_mean, 'average')
            


#################################################### SP
    for i in range(n_c):
        
        #for ii in range(3):
        #    if ii == 0:
        #        indices = sea_indices
        #    elif ii == 1:
        #        indices = plain_indices
        #    elif ii == 2:
        #        indices = mountain_indices
            fig,axs = plt.subplots(figsize = (9,7))
            
            for k in range(len_tests):
                    
                #plt.ylim([-0.3, 0.3])
                #print(s_p_scores_LT[k,:,:,:,i, indices[:,0], indices[:,1]].shape)
                #plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i, indices[:,0], indices[:,1]]**2., axis =(0,1)))/np.nanmean(s_p_scores_LT[k,:,:,1,i, indices[:,0], indices[:,1]], axis =(0,1)), label = tests_list[k], color= color_p[k] )
                #plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i, indices[:,0], indices[:,1]]**2., axis =(0,1))), label = tests_list[k], color= color_p[k] )
                plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i]**2., axis =(0,2,3))), label = cases_clean[k], color= color_p[k] )
                #plt.plot(np.nanmean(s_p_scores_LT[k,:,:,1,i], axis =(0,2,3))/np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i]**2., axis =(0,2,3))), label = cases_clean[k], color= color_p[k] )

                plt.plot(np.nanmean(s_p_scores_LT[k,:,:,1,i], axis =(0,2,3)), linestyle='dashed', color= color_p[k] )
                plt.xticks( fontsize ='18')
                axs.set_xticks(range(len(echeance)))
                axs.set_xticklabels(echeance)
                axs.tick_params(direction='in', length=12, width=2)
                plt.yticks(fontsize ='18')
                #plt.title(var_names[i] + ' ' + domain[ii],fontdict = font)
                plt.title(var_names[i],fontdict = font)
                #plt.text(0.6, 0.9, var_names[i] + ' ' + domain[ii],
                  #        fontdict = font, transform=axs.transAxes)
                plt.legend(fontsize = 14,frameon = False, ncol=2)
                plt.savefig('/scratch/mrmn/moldovang/score_ensemble/plots/skill_spread/skill_spread'+str(i) +'.png')
            




################################################"" BRIER




    for i in range(6):
        
        for j in range(n_c):
            fig,axs = plt.subplots(figsize = (9,7))
            for k in range(len_tests):
            
                #plt.plot(1-np.nanmean(Brier_scores_LT[k,:,:,i, j], axis= (0,2,3))/np.nanmean(Brier_scores_LT[0,:,:,i, j], axis= (0,2,3)), label = cases_clean[k], color = color_p[k])
                plt.plot(np.nanmean(Brier_scores_LT[0,:,:,i, j], axis= (1,2,3))-np.nanmean(Brier_scores_LT[k,:,:,i, j], axis= (1,2,3)), label = cases_clean[k], color = color_p[k])
                
                # Brier_s = np.nanmean(Brier_scores_LT[0,:,0,i, j], axis= (1,2))-np.nanmean(Brier_scores_LT[k,:,0,i, j], axis= (1,2))
                # counter=0
                # for ii in range(74):
                #     if Brier_s[ii]<0:
                        
                #         counter = counter+1
                
                # print('Test is ' ,k, 'variable is',  j, 'threshold is', i, 'counter is', counter)
                        
            #plt.ylim([-0.15, 0.15])
            plt.xticks( fontsize ='18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[j][i],fontdict = font)
            #plt.text(0.3, 0.9, case_name[j][i],
            #         fontdict = font, transform=axs.transAxes)
            plt.legend(fontsize = 14,frameon = False, ncol=2)
            plt.savefig('/scratch/mrmn/moldovang/score_ensemble/plots/brier/brier'+str(i)+'_'+str(j)+'.png')
            

#####################################################"ROC

    #bins_roc = np.array([0.01, 0.07, 0.14, 0.23, 0.3, 0.37, 0.44, 0.51, 0.58, 0.65, 0.72, 0.79, 0.86, 0.93, 0.99])
    bins_roc = np.array([0.99, 0.93, 0.86, 0.79, 0.72, 0.65, 0.58, 0.51, 0.44, 0.37, 0.3, 0.23, 0.14, 0.07, 0.01])
    for i in range(6):

        for j in range(n_c):
            fig,axs = plt.subplots(figsize = (9,7))
            for k in range(len_tests):
                O_tr = rel_diag_scores[k,:-2,i,1,j]
                X_prob = rel_diag_scores[k,:-2,i,0,j]
                #print(O_tr.shape, X_prob.shape,)

                for z in range(bins_roc.shape[0]):
                    
                    forecast_p = copy.deepcopy(X_prob[np.where((X_prob > bins_roc[z]), True, False)])
                    obs = copy.deepcopy(O_tr[np.where((X_prob > bins_roc[z]), True, False)])
                    obs_w_nan = copy.deepcopy(obs[~np.isnan(obs)])
                    for_w_nan = copy.deepcopy(forecast_p[~np.isnan(obs)])
                    for_w_nan[:] = 1
                    TP = (for_w_nan == obs_w_nan).sum()
                    FP = (for_w_nan != obs_w_nan).sum()
                    
                    
                    forecast_n = copy.deepcopy(X_prob[np.where((X_prob <= bins_roc[z]), True, False)])
                    obs = copy.deepcopy(O_tr[np.where((X_prob <= bins_roc[z]), True, False)])
                    obs_w_nan = copy.deepcopy(obs[~np.isnan(obs)])
                    for_w_nan = copy.deepcopy(forecast_n[~np.isnan(obs)])
                    for_w_nan[:] = 0
                    TN = (for_w_nan == obs_w_nan).sum()
                    FN = (for_w_nan != obs_w_nan).sum() 
                    
                    Hit_rate[z+1]= (TP/(TP+FN))
                    false_alarm[z+1] = (FP/(FP+TN))
                    
                #print(np.trapz(Hit_rate, false_alarm))
                    #freq_obs[z] = obs.sum()/obs.shape[0]
                plt.plot(false_alarm, Hit_rate, label = cases_clean[k], color = color_p[k])
                A_ROC[k] = np.trapz(Hit_rate, false_alarm)
                    
                A_ROC_skill[k]=1-A_ROC[0]/A_ROC[k]
                

            #plt.plot(bins[:-1]+0.05, bins[:-1]+0.05, label = 'perfect', color = 'black')
                #plt.ylim([-0.15, 0.15])
            plt.xticks( fontsize ='18')
            plt.xlabel('False Alarm Rate', fontsize= '18')
            plt.ylabel('Hit Rate', fontsize= '18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[j][i],fontdict = font)
            #plt.text(0.3, 0.9, case_name[j][i],
            #         fontdict = font, transform=axs.transAxes)
            plt.legend(fontsize = 14,frameon = False, ncol=1)
            
            #axins = inset_axes(axs, width="60%", height="30%", loc=8)

            #axins.bar(tests_list[1::], A_ROC_skill[1::])

            plt.savefig('/scratch/mrmn/moldovang/score_ensemble/plots/ROC/ROC'+str(i)+'_'+str(j)+'.png')
            
            fig,axs = plt.subplots(figsize = (9,7))
            
            fig,axs = plt.subplots(figsize = (9,7))
            plt.bar(cases_clean[1::], A_ROC_skill[1::])
            plt.xticks( fontsize ='18')
            plt.ylabel('Area under ROC skill', fontsize= '18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[j][i],fontdict = font)
            
            plt.savefig('/scratch/mrmn/moldovang/score_ensemble/plots/ROC/AROC'+str(i)+'_'+str(j)+'.png')


Path_to_q = '/scratch/mrmn/moldovang/tests_CGAN/'
#tests_list = ['REAL', 'INVERSION', 's_w_p_F', 'interp_alpha_1.5', 'sm_2_4_9_12_W', 'sm_0_2_9_12_W', 'sm_3_4_9_12_W' ]
#tests_list = ['REAL', 'sm_4_12_W', 'sm_0_2_9_12_W', 'sm_0_3_9_12_W', 'sm_1_3_9_12_W', 'sm_2_3_9_12_W',
              #'sm_2_4_9_12_W', 'sm_3_4_9_12_W']
              
#tests_list = ['REAL', 'sm_2_3_9_12_W', 's_w_p_F', 'interp_alpha_1.5', 'interp_alpha_1.25', 'sm_0_2_9_12_W',
#              'sm_4_12_W', 'sm_2_4_9_12_W', 'sm_3_4_9_12_W']

tests_list = ['REAL', 'sm_2_3', 'sm_2_3_9_12_W', 's_w_p_F', 'interp_alpha_1.5', 'sm_0_2_9_12_W' ]
tests_list = ['REAL', 'sm_0_1', 'sm_1_2', 'sm_2_3', 'sm_4_5', 'interp_alpha_1.5', 's_w_p_F', 'sm_2_3_10_12_W', 's_w_p_F', 'interp_alpha_1.25', 'interp_alpha_1.5',
              'sm_3_4_9_12_W', 'sm_3_4_10_12_W', 'sm_2_4_9_12_W', 'sm_2_4_10_12_W', 'sm_0_3_9_12_W', 'sm_1_3_9_12_W', 'sm_2_3_9_12_W']

tests_list = ['REAL', 'sm_0_2_9_12_W', 'sm_2_3_10_12_W', 's_w_p_F', 'interp_alpha_1.5']
tests_list = ['REAL', 'sm_2_3_10_12_W_200iter', 's_w_p_F', 'interp_alpha_1.5']
tests_list = ['REAL', 'sm_2_3_10_12_W', 'sm_2_3_10_12_W_200_iter_0.1', 'sm_2_3_10_12_W_300_iter_noise', 'sm_2_3_10_12_W_300_iter_noise_zero', 'sm_2_3_10_12_W_300_iter_noise_random' ]
tests_list = ['REAL', 'sm_2_3_10_12_W', 'sm_2_3_10_12_W_200_iter_0.1', 'sm_2_3_10_12_W_300_iter_0.1', 'sm_2_3_10_12_W_400_iter_0.1', 'sm_2_3_10_12_W_300_iter_noises_0.1']
#tests_list = ['REAL', 'sm_2_3_10_12_W', 'sm_2_3_10_12_W_1000_iter_0.1',]
tests_list = ['REAL', 'sm_2_3_10_12_W', 'sm_2_3_10_12_W_400_iter_0.1_mse', 'sm_2_3_10_12_W_400_iter_noises_0.1_mse', 'sm_2_3_10_12_W_400_iter_0.25_mse' ]
tests_list = ['REAL', 'sm_2_3_10_12_W', 'sm_2_3_10_12_W_400_iter_0.25_mse', 'sm_2_3_10_12_W_400_iter_noises_0.25_mse', 'sm_2_3_10_12_W_400_iter_noises_0.25_mse_nseon' ]

tests_list = ['REAL', 'sm_2_3_10_12_W_inv_200_iter_noises_0.1_mse', 'sm_2_3_10_12_W_inv_200_iter_noises_0.1_mse_nseon',
              'sm_2_3_10_12_W_inv_400_iter_noises_0.1_mse', 'sm_2_3_10_12_W_inv_400_iter_noises_0.1_mse_nseon', 'sm_2_3_10_12_W_inv_400_iter_0.1_mse',
              'sm_2_3_10_12_W_inv_400_iter_noises_0.25_mse', 'sm_2_3_10_12_W_inv_400_iter_noises_0.25_mse_nseon',
              'sm_2_3_10_12_W_inv_400_iter_noises_0.1_l1', 'sm_2_3_10_12_W_inv_400_iter_noises_0.1_l1_nseon',
              'sm_2_3_10_12_W_inv_1000_iter_0.1_mse'] # INVERSION
tests_list = ['REAL', 'sm_2_3_10_12_W_200_iter_noises_0.1_mse', 'sm_2_3_10_12_W_200_iter_noises_0.1_mse_nseon',
              'sm_2_3_10_12_W_400_iter_noises_0.1_mse', 'sm_2_3_10_12_W_400_iter_noises_0.1_mse', 'sm_2_3_10_12_W_400_iter_0.1_mse',
              'sm_2_3_10_12_W_400_iter_noises_0.25_mse', 'sm_2_3_10_12_W_400_iter_noises_0.25_mse_nseon',
              'sm_2_3_10_12_W_400_iter_0.1', 'sm_2_3_10_12_W_400_iter_0.1',
              'sm_2_3_10_12_W'] # INVERSION
tests_list = ['REAL','sm_2_3_10_12_W_1000_iter_OPT_1', 'sm_2_3_10_12_W_1000_iter_OPT_5', 'sm_2_3_10_12_W_1000_iter_OPT_10', 'sm_2_3_10_12_W_1000_iter_OPT_20']
#tests_list = ['REAL', 'from_z_w_plus_False_512', 's_w_p_F']
#tests_list = ['REAL', 'sm_4_12_W_c', 'sm_5_12_W_c', 'sm_6_12_W_c', 'sm_5_12_W_O', 'sm_5_12_W_On', 'sm_5_12_W_O0', 'sm_5_12_W_O1.5'  ]
tests_list = ['REAL','sm_2_3_10_12_W_1000_iter_OPT_20', 'sm_2_4_10_12_W_1000_iter_OPT_0.025_20', 'sm_2_4_10_12_W',  'sm_1_3_10_12_W_1000_iter_OPT_0.025_20']

tests_list = ['REAL','sm_2_3_10_12_W_1000_iter_OPT_40', 'sm_2_4_10_12_W_1000_iter_OPT_0.025_40', 'sm_1_4_10_12_W_1000_iter_OPT_0.025_40']
tests_list = ['REAL','sm_2_4_10_12_W','sm_2_4_10_12_W_1000_iter_OPT_0.025_20', 'sm_2_4_10_12_W_1000_iter_OPT_0.025_40']

tests_list = ['REAL','sm_2_3_10_12_W_1000_iter_OPT_1', 'sm_2_3_10_12_W', 'sm_2_3_10_12_W_1000_iter_OPT_5', 'sm_2_3_10_12_W_1000_iter_OPT_10', 'sm_2_3_10_12_W_1000_iter_OPT_20', 'sm_2_3_10_12_W_1000_iter_OPT_40']
tests_list = ['REAL', 'throughB_0_1_1_12_1000_0_0.1_0.001_1.0_50_30_10_freeNoise', 'throughB_2_3_10_12_1000_0_0.1_0.001_1.0_50_30_10_freeNoise']
tests_list = ['REAL', 'throughB_0_1_1_12_1000_0_0.1_0.001_1.0_50_30_10_freeNoise', 'throughB_2_3_10_12_1000_0_0.1_0.001_1.0_50_30_10_freeNoise']
tests_list = ['REAL', 'sm_2_3_10_12_W_300_iter_noise', 'sm_2_3_10_12_W_300_iter_noises_0.1']

plots_ens(tests_list, Path_to_q, 7, 594, 3, 128)
