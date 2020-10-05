'''
plot the loss functions of the various algorithms

'''

import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 15})
import os



### EVALUATING LOSS FUNCTION PERFORMANCE #### 

current_dir = os.getcwd()
target_dir_1 = current_dir + '/saved_models/SecEncDec_4in_2out_Run1_6_1/'
target_dir_2 = current_dir + '/saved_models/ATTN_4in_2out_Run2_6_4/'
# target_dir_3 = current_dir + '/saved_models/ConvEncDecRe_100_sims_10_epochs_test/'

model_comp = [target_dir_1,target_dir_2]#,target_dir_3]
loss_vecs = [torch.load(folder+'tr_loss.pth') for folder in model_comp]
val_loss_vecs = [torch.load(folder+'val_loss.pth') for folder in model_comp]

for j in range(len(model_comp)):
    smoothed_loss = [np.mean(loss_vecs[j][i-100:i]) for i in range(100,len(loss_vecs[j]))]
    smoothed_val_loss = [np.mean(val_loss_vecs[j][i-100:i]) for i in range(100,len(val_loss_vecs[j]))]

    plt.plot(smoothed_val_loss)
    plt.yscale('log')
    plt.ylim([0,1])
    plt.ylabel('Loss')
    plt.xlabel('Number of updates')
    plt.title('2 Step Validation Loss Comparison')
plt.legend(['Multi-Input FF','Attention'])
plt.show()

### Evaluating Accuracy over time ###

current_dir = os.getcwd()
target_dir_1 = current_dir + '/model_comparison/ATTN_4_2.pt'
target_dir_2 = current_dir + '/model_comparison/Seq_Conv_Encoder_4_2.pt'
target_dir_3 = current_dir + '/model_comparison/SISO.pt'

model_comp = [target_dir_1,target_dir_2,target_dir_3]
error_matrices = [torch.load(files) for files in model_comp]

for i in range(len(model_comp)):
    plt.plot(error_matrices[i][:200])
    plt.legend(['Attn','Multi','Single'])
    plt.title('Error Over Time')
    plt.ylabel('Mean Error')
    plt.xlabel('Time Steps')

plt.show()

for i in range(len(model_comp)):
    print(error_matrices[i][0],
          error_matrices[i][9],
          error_matrices[i][49],
          error_matrices[i][99])

