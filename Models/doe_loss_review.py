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
current_dir = os.getcwd()
target_dir = current_dir + '/tune/old_does/'
legend = []
plt.figure(figsize = (8,5))

# first_dir = '/saved_models/ConvEncDecRe_100_sims_10_epochs_test/'
# first_file = current_dir + first_dir + 'tr_loss.pth'

# loss = torch.load(first_file)
# smoothed_loss = [np.mean(loss[i-100:i]) for i in range(100,len(loss))]
# plt.plot(smoothed_loss[:800])
# legend.extend([str(1)]) 

for i in range(2,11):
    # if i != 6:
    #     if i == 1:
    #         file_path = target_dir + 'ConvEncDecRe_100_sims_10_epochs_test/tr_loss.pth'
    #         loss = torch.load(file_path)
    #         smoothed_loss = [np.mean(loss[i-50:i]) for i in range(10,len(loss))]
    #         plt.plot(smoothed_loss)
    #         legend.extend([str(i)])
    #     else:
            file_path = target_dir + 'ATTN_'+ str(i) + \
                        '_100_sims_quarter_epoch_doe/tr_loss.pth'
            loss = torch.load(file_path)
            smoothed_loss = [np.mean(loss[i-100:i]) \
                            for i in range(100,len(loss))]
            plt.plot(smoothed_loss[:800])
            legend.extend([str(i)]) 

plt.yscale('log')
plt.legend(legend)
plt.xlim([0,1000])
plt.ylim([0,.1])
plt.title('Number of Inputs to Attention Model')
plt.ylabel('Loss')
plt.xlabel('Number of updates')
plt.show()