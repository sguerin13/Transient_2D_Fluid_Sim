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
target_dir = current_dir + '/tune/attn_doe_2_5_31/'
legend = []
plt.figure(figsize = (15,10))

# visualize the loss ranges
for files in os.listdir(target_dir):

    file_str = target_dir + files
    model_dict = torch.load(file_str)
    loss = model_dict['loss']
    smoothed_loss = [np.mean(loss[i-100:i]) for i in range(10,len(loss))]
    plt.plot(smoothed_loss)


plt.yscale('log')
plt.xlim([0,600])
plt.ylim([0,1])
plt.title('Number of Inputs to FF Model - 100-Step Smoothing')
plt.ylabel('Loss')
plt.xlabel('Number of updates')
plt.show()

# loss scatter plot
for files in os.listdir(target_dir):

    file_str = target_dir + files
    model_dict = torch.load(file_str)
    loss = model_dict['loss']
    loss_mean = np.mean(loss[-100:])
    plt.scatter(model_dict['lr'],loss_mean)

    
plt.yscale('log')
plt.xscale('log')
plt.ylim([0,1])
plt.title('Loss vs Lr')
plt.ylabel('Loss')
plt.xlabel('LR')
plt.show()


# loss scatter plot
for files in os.listdir(target_dir):

    file_str = target_dir + files
    model_dict = torch.load(file_str)
    loss = model_dict['loss']
    loss_mean = np.mean(loss[-100:])
    plt.scatter(model_dict['batch'],loss_mean)

    
plt.yscale('log')
plt.xscale('log')
plt.ylim([0,1])
plt.title('Loss vs Batch')
plt.ylabel('Loss')
plt.xlabel('Batch_Size')
plt.show()


# loss scatter plot
for files in os.listdir(target_dir):

    file_str = target_dir + files
    model_dict = torch.load(file_str)
    loss = model_dict['loss']
    loss_mean = np.mean(loss[-100:])
    plt.scatter(model_dict['seq_x'],loss_mean)
    
plt.yscale('log')
plt.ylim([0,1])
plt.title('Loss vs seq_x')
plt.ylabel('Loss')
plt.xlabel('Input Length')
plt.show()


# loss scatter plot
for files in os.listdir(target_dir):

    file_str = target_dir + files
    model_dict = torch.load(file_str)
    loss = model_dict['loss']
    loss_mean = np.mean(loss[-100:])
    plt.scatter(model_dict['seq_y'],loss_mean)
    
plt.yscale('log')
plt.ylim([0,1])
plt.title('Loss vs seq_y')
plt.ylabel('Loss')
plt.xlabel('output Length')
plt.show()

min_loss = 1
# loss scatter plot
for files in os.listdir(target_dir):

    file_str = target_dir + files
    model_dict = torch.load(file_str)
    loss = model_dict['loss']
    loss_mean = np.mean(loss[-100:])
    if loss_mean < min_loss:
        min_loss = loss_mean
        min_lr   = model_dict['lr']
    plt.scatter(model_dict['seq_y'],loss_mean)