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
target_dir = current_dir + '/saved_models/SecEncDec_4in_2out_Run1_6_1/'
loss  = torch.load(target_dir+'tr_loss.pth')
val_loss = torch.load(target_dir+'val_loss.pth')
smoothed_loss = [np.mean(loss[i-100:i]) for i in range(100,len(loss))]
smoothed_val_loss = [np.mean(val_loss[i-100:i]) for i in range(100,len(val_loss))]

plt.plot(smoothed_loss)
plt.yscale('log')
plt.ylim([0,1])
plt.ylabel('Loss')
plt.xlabel('Number of updates')
plt.title('FF Model  - 100-Step Smoothing - training')
plt.show()

plt.plot(smoothed_val_loss)
plt.yscale('log')
plt.ylim([0,1])
plt.ylabel('Loss')
plt.xlabel('Number of Updates')
plt.title('FF Model  - 100-Step Smoothing - validation')
plt.show()