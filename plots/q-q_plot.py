
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pickle
from torchvision import transforms




def qq_plots(tests_list, Path_to_q, n_q, N_e, n_c, size_x):
    
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
    print(sea_indices.shape, mountain_indices.shape, plain_indices.shape)
    
    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 25,
        }
    len_tests = len(tests_list)
    name_res = '/log/qtiles_standalone_metrics_592.p'
    Quantiles=np.zeros((len_tests, N_e, n_q, n_c, size_x, size_x), dtype = 'float32')
    
    for i in range(len_tests):
        
        quant = pickle.load(open(Path_to_q + tests_list[i] + name_res, 'rb'))
        quant = quant['quantiles']
        Quantiles[i] = quant
        
    
        

    color_p = ['darkgreen', 'royalblue', 'purple', 'darkorange', 'red', 'cyan', 'gold', 'black']
    var = ['u (m/s)', 'v (m/s)', 't2m (K)']
    zone = ['sea', 'mountain', 'plain']
    for i in range(3):
        for j in range(3):
            fig,axs = plt.subplots(figsize = (9,7))
            for k in range(len_tests):
                
                
                Quantiles_avg_d = np.zeros((3, n_q, n_c))
                Quantiles_avg = Quantiles[k, 0:(N_e-2), :, :, :, :].mean(axis=0)
                Quantiles_avg_d[0] = Quantiles_avg[:,:, sea_indices[:,0], sea_indices[:,1]].mean(axis=2)
                Quantiles_avg_d[1] = Quantiles_avg[:,:, mountain_indices[:,0], mountain_indices[:,1]].mean(axis=2)
                Quantiles_avg_d[2] = Quantiles_avg[:,:, plain_indices[:,0], plain_indices[:,1]].mean(axis=2)
                if k == 0:
                    Quantiles_avg_d_REAL = Quantiles_avg_d
                    
                plt.plot((Quantiles_avg_d_REAL[i,:,j]*(1/0.95)*Maxs[j].item()+Means[j].item()), (Quantiles_avg_d[i,:,j]*(1/0.95)*Maxs[j].item()+Means[j].item()),
                         linestyle = 'dashdot', marker='o',  color = color_p[k], label  = tests_list[k])
                
                #plt.title(var_names[j], fontsize = 25)        
                #plt.grid()
             
            #plt.xticks([0.5,1.0,2.0],[r'$10^{0.5}$', r'$10^{1}$', r'$10^{2}$'], fontsize ='22')
            plt.xticks( fontsize ='18')
            axs.tick_params(direction='in', length=12, width=2)
                
                
            plt.yticks(fontsize ='18')
            #plt.xlabel(r' $Q_{AROME}$', fontsize = '25',fontweight ='bold')
            plt.xlabel(var[j], fontdict = font)
            plt.legend(fontsize = 16,frameon = False)
            #plt.ylabel(r'$Q_{GAN}$', fontsize = 25)
            plt.ylabel(var[j], fontdict = font)
            plt.text(0.7, 0.1, zone[i],fontdict = font, transform=axs.transAxes)
            #plt.text(0.55, 0.6, "spam", size=50, rotation=-25.,ha="right", va="top",
            #         bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
            
            fig.tight_layout()
            plt.savefig('QQ_plots'+str(i)+'_'+str(j)+'.png')
            #plt.savefig(output_dir+'Spectral_PSD_{}.png'.format(var_names[j]))
            plt.close()
#####################################################################


Path_to_q = '/scratch/mrmn/moldovang/tests_CGAN/'
tests_list = ['REAL', 'INVERSION', 's_w_p_F', 'interp_alpha_1.5', 'sm_2_4_9_12_W', 'sm_0_2_9_12_W', 'sm_3_4_9_12_W' ]
#tests_list = ['REAL', 'sm_4_12_W', 'sm_0_2_9_12_W', 'sm_0_3_9_12_W', 'sm_1_3_9_12_W', 'sm_2_3_9_12_W',
              #'sm_2_4_9_12_W', 'sm_3_4_9_12_W']
qq_plots(tests_list, Path_to_q, 7, 594, 3, 128)