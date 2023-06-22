
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pickle
from torchvision import transforms
import copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



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
    
    
    color_p = ['red', 'royalblue', 'darkgreen', 'darkorange', 'red', 'cyan', 'gold', 'black']

    case_name = [['ff=3 (m/s)', 'ff=4 (m/s)', 'ff=5 (m/s)', 'ff=6 (m/s)', 'ff=7 (m/s)', 'ff=8 (m/s)'],
                 ['', '', '', '', '', ''], 
                 ['t2m=278.15 (K)', 't2m=281.15 (K)', 't2m=283.15 (K)', 't2m=285.15 (K)', 't2m=287.15 (K)', 't2m=289.15 (K)']]

    case_name = [['ff=5 (m/s)', 'ff=7.5 (m/s)', 'ff=10 (m/s)', 'ff=12.5 (m/s)', 'ff=15 (m/s)', 'ff=17.5 (m/s)'],
                 ['', '', '', '', '', ''], 
                 ['t2m=278.15 (K)', 't2m=283.15 (K)', 't2m=288.15 (K)', 't2m=293.15 (K)', 't2m=298.15 (K)', 't2m=303.15 (K)']]


    var_names = ['ff', 'dd', 't2m']
    var_names_m = ['ff (m/s)', 'dd (°)', 't2m (K)'  ]
    domain = ['sea', 'plain', 'mountain']
    cases_clean = ['AROME', 'Style Perturbation', 'PCA Resampling', 'Extrapolation']
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
            
                plt.plot(1-np.nanmean(Brier_scores_LT[k,:,:,i, j], axis= (0,2,3))/np.nanmean(Brier_scores_LT[0,:,:,i, j], axis= (0,2,3)), label = cases_clean[k], color = color_p[k])
            
            #plt.ylim([-0.15, 0.15])
            plt.xticks( fontsize ='18')
            axs.tick_params(direction='in', length=12, width=2)
            plt.yticks(fontsize ='18')
            plt.title(case_name[j][i],fontdict = font)
            #plt.text(0.3, 0.9, case_name[j][i],
            #         fontdict = font, transform=axs.transAxes)
            plt.legend(fontsize = 14,frameon = False, ncol=2)
            plt.savefig('/scratch/mrmn/moldovang/score_ensemble/plots/brier/brier'+str(i)+'_'+str(j)+'.png')
            

################


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
tests_list = ['REAL', 'sm_2_3_10_12_W', 's_w_p_F', 'interp_alpha_1.5']

#tests_list = ['REAL', 'sm_4_12_W_c', 'sm_5_12_W_c', 'sm_6_12_W_c', 'sm_5_12_W_O', 'sm_5_12_W_On', 'sm_5_12_W_O0', 'sm_5_12_W_O1.5'  ]
plots_ens(tests_list, Path_to_q, 7, 594, 3, 128)
