
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


def plots_ens(tests_list, Path_to_q, n_q, N_e, n_c, size_x, n_LT, n_D):
    

    # plain_indices = np.array(plain_indices)

    
    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 25,
        }
    len_tests = len(tests_list)
    name_res = '/log/distance_metrics_distance_metrics_300.p'

    name_res_rd = '/log/rel_diagram_distance_metrics_592.p'
    N_bins_max = 121

    
    bias = np.zeros((len_tests, N_e, n_c, size_x, size_x), dtype = ('float32'))
    bias_LT = np.zeros((len_tests, n_D, n_LT, n_c, size_x, size_x), dtype = ('float32'))
    mse = np.zeros((len_tests, N_e, n_c, size_x, size_x), dtype = ('float32'))
    mse_LT = np.zeros((len_tests, n_D, n_LT, n_c, size_x, size_x), dtype = ('float32'))


    
    

    for i in range(len_tests):
        

        bias_load = np.load(Path_to_q + tests_list[i] + '/log/distance_metrics_distance_metrics_2265_bias.npy')
        mse_load = np.load(Path_to_q + tests_list[i] + '/log/distance_metrics_distance_metrics_2265_mse.npy')

        bias[i] = bias_load
        mse[i] = mse_load
        
    """
    Transformation to identify Lead Times more easily

    """
    D_i = 0
    LT_i = 0
    for i in range(N_e-2):        
        
        bias_LT[:,D_i, LT_i] = bias[:,i]
        mse_LT[:,D_i, LT_i] = mse[:,i]
        LT_i =LT_i + 1
        
        if LT_i == n_LT : 
            
            D_i = D_i +1
            LT_i = 0
        
        
    #############################################################################################
    
    
    color_p = ['black', 'royalblue', 'darkgreen', 'darkorange', 'red', 'cyan', 'gold', 'pink', 'tan', 'slategray', 'purple', 'palegreen', 'orchid', 'crimson', 'firebrick']

    case_name = [['ff=3 (m/s)', 'ff=4 (m/s)', 'ff=5 (m/s)', 'ff=6 (m/s)', 'ff=7 (m/s)', 'ff=8 (m/s)'],
                 ['', '', '', '', '', ''], 
                 ['t2m=278.15 (K)', 't2m=281.15 (K)', 't2m=283.15 (K)', 't2m=285.15 (K)', 't2m=287.15 (K)', 't2m=289.15 (K)']]

    case_name = [['ff=5 (m/s)', 'ff=7.5 (m/s)', 'ff=10 (m/s)', 'ff=12.5 (m/s)', 'ff=15 (m/s)', 'ff=17.5 (m/s)'],
                 ['', '', '', '', '', ''], 
                 ['t2m=278.15 (K)', 't2m=283.15 (K)', 't2m=288.15 (K)', 't2m=293.15 (K)', 't2m=298.15 (K)', 't2m=303.15 (K)']]

    remove_comm = 'rm -r -f ' + 'brier crps ROC rel_diag mean_bias skill_spread rank_histo bias mse'
    os.system(remove_comm)
    os.mkdir('brier') 
    os.mkdir('crps') 
    os.mkdir('ROC') 
    os.mkdir('rel_diag')
    os.mkdir('mean_bias')
    os.mkdir('skill_spread')
    os.mkdir('rank_histo')
    os.mkdir('mse')
    os.mkdir('bias')
    
    PATH_SAVE = '/home/mrmn/moldovang/score_ensemble/plots/'
    var_names = ['ff', 'dd', 't2m']
    var_names_m = ['ff (m/s)', 'dd (°)', 't2m (K)'  ]
    domain = ['sea', 'plain', 'mountain']

    cases_clean = ['AROME','200', '400', '600', '800', '1000']

    #cases_clean = ['AROME', 'from_z', 's_w_p_f']
    #cases_clean = ['AROME', '1000', '200', '300_N', '300_Z', '300_R']
    echeance = ['3H', '6H', '9H', '12H', '15H', '18H', '21H', '24H', '27H', '30H', '33H', '36H', '39H', '42H', '45H']
    echeance = ['3H', '', '9H', '', '15H', '', '21H', '', '27H', '', '33H', '', '39H', '', '45H']

################################################ MEAN BIAS    
    for i in range(n_c):
        

        fig,axs = plt.subplots(figsize = (9,7))
        
        for k in range(len_tests):
                
            #plt.ylim([-0.3, 0.3])
            #plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i, indices[:,0], indices[:,1]]**2., axis =(0,1)))/np.nanmean(s_p_scores_LT[k,:,:,1,i, indices[:,0], indices[:,1]], axis =(0,1)), label = tests_list[k], color= color_p[k] )
            #plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i, indices[:,0], indices[:,1]]**2., axis =(0,1))), label = tests_list[k], color= color_p[k] )
            plt.plot(np.mean(bias_LT[k,:,:,i], axis = (0,2,3)), label = cases_clean[k], color= color_p[k] )


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
            plt.savefig(PATH_SAVE+'bias/bias'+str(i)+'.png')
        
    for i in range(n_c):
        

        fig,axs = plt.subplots(figsize = (9,7))
        
        for k in range(len_tests):
                
            #plt.ylim([-0.3, 0.3])
            #plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i, indices[:,0], indices[:,1]]**2., axis =(0,1)))/np.nanmean(s_p_scores_LT[k,:,:,1,i, indices[:,0], indices[:,1]], axis =(0,1)), label = tests_list[k], color= color_p[k] )
            #plt.plot(np.sqrt(np.nanmean(s_p_scores_LT[k,:,:,0,i, indices[:,0], indices[:,1]]**2., axis =(0,1))), label = tests_list[k], color= color_p[k] )
            plt.plot(np.sqrt(np.mean(mse_LT[k,:,:,i], axis = (0,2,3))), label = cases_clean[k], color= color_p[k] )


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
            plt.savefig(PATH_SAVE+'mse/mse'+str(i)+'.png')



    
    


Path_to_q = '/scratch/mrmn/moldovang/tests_CGAN/'
tests_list = ['REAL_256', 'INVERSION_200', 'INVERSION_400', 'INVERSION_600', 'INVERSION_800', 'INVERSION_1000']


plots_ens(tests_list, Path_to_q, 7, 2267, 3, 256, 15, 151)
