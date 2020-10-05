'''
method to load and run the model with a model that uses multiple steps as
inputs. Also saves an animation. 
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler,DataLoader,TensorDataset,random_split
from SeqConvEncoderDecoder import Decoder, Encoder, EncoderDecoder
from ConvAttnModel import ConvAttn
from Modules.mask_loss import MaskedLoss
from Modules.BigDataSet import BigAssDataset
from Modules.BigDataSetR2 import XYDataset
from Utils.data_prep import concat_collate
import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from matplotlib import animation
import os

# state_dict_path = os.getcwd()+'/saved_models/ATTN_4in_2out_Run1_5_31/ATTN_4in_2out_Run1_5_31.pth'
state_dict_path = os.getcwd()+'/saved_models/SecEncDec_4in_2out_Run1_6_1/SecEncDec_4in_2out_Run1_6_1.pth'
data_file_path   = os.getcwd() + '/../../Data/data/100sims/seqxy/x4y/'

# load model
model = EncoderDecoder(num_scenes = 4).float()
# model = ConvAttn(seq_length = 4,n_heads=4,model_dropout=0,device = 'gpu').float()
model.load_state_dict(torch.load(state_dict_path))
model.cuda().float()

torch.set_grad_enabled(False)
model.eval()

# load file paths
files = os.listdir(data_file_path)
# may need to load scalers and unscale

def load_sample(data): 
    # send data to gpu
    x    = data[0].cuda().float()
    x = x.unsqueeze(0)
    Re   = data[1].cuda().float()
    Re = Re.unsqueeze(0)
    mask = data[2].cuda().float()
    y    = data[3].cuda().float()
    # print(x.shape,Re.shape,mask.shape,y.shape)
    mask = torch.unsqueeze(mask,dim=1)

    return x,Re,mask,y

def pass_through_model(model,x,Re):
    out = model.forward(x,Re)
    out_numpy = out.cpu().detach().numpy()
    # print(out_numpy.shape)
    return out, out_numpy

def vorticity(x,y):
    vort = np.abs(x[1:-1, 2:] - x[1:-1, :-2]
                - y[2:, 1:-1] + y[:-2, 1:-1])
    return vort

def animate_results(out,y,mask,im_list,metric='vorticity'):
    out_masked = out*mask.cpu().detach().numpy()
    out_x = out_masked[0,0,:,:]
    out_y = out_masked[0,1,:,:]
    y_masked   = y.detach().cpu().numpy()*mask.cpu().detach().numpy()
    # print(y_masked.shape)
    y_x = y_masked[0,0,:,:]
    y_y = y_masked[0,1,:,:]

    if metric == 'vx':
        im_list.append([y_x,out_x,y_x-out_x])

    if metric == 'vy':
        im_list.append([y_y,out_y,y_y-out_y])

    if metric == 'vorticity':
        v_gt = vorticity(y_x,y_y)
        v_pred = vorticity(out_x,out_y)
        im_list.append([v_gt,v_pred,v_gt-v_pred])

    return im_list

def main(loss_steps = 1,multi_in = True,metric = 'vx'):
    im_list = []
    for i in range(77,78):

        for j in range(240):
            file_str = '100sims_sim_'+str(i)+'sample_'+str(j)+'.pt'
            data = torch.load(data_file_path + file_str)

            if j == 0:
                
                x,Re,mask,y = load_sample(data,loss_steps=1)
                print(Re)
                out,out_numpy = pass_through_model(model,x,Re)
                # print(out.shape)
                im_list = animate_results(out_numpy,y,
                                            mask,im_list,
                                            metric=metric)                                            
                # update the input with the output from the model
                x[:,-1,:-1,:,:] = out
                del out,out_numpy
                
            else:
                _,_,_,y = load_sample(data,loss_steps=1)
                out,out_numpy = pass_through_model(model,x,Re)
                im_list = animate_results(out_numpy,y,
                                            mask,im_list,
                                            metric=metric
                                            )
                # update the input with the output from the model
                x[:,-1,:-1,:,:] = out
                del out,out_numpy
               
    return  im_list

im_list = main(loss_steps = 1,multi_in = True,metric = 'vx')

import matplotlib.animation as anim
import matplotlib as mpl
from matplotlib.colors import  BoundaryNorm
fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0.06, 0.44, 0.0125, 0.1])

ax1=fig.add_subplot(1,3,1)
norm = mpl.colors.Normalize(vmin=-3, vmax=3)
cbar = mpl.colorbar.ColorbarBase(ax, cmap=mpl.cm.bwr, norm=norm,
                                orientation='vertical')
                                
ax.yaxis.set_ticks_position("left")
# plot = ax1.pcolor([-3,3])
# fig.colorbar(plot)
ax1.set_title('Ground Truth')
ax2=fig.add_subplot(1,3,2)
ax2.set_title('Predicted')
ax3=fig.add_subplot(1,3,3)
ax3.set_title('Difference')

ims = []
for i in range(len(im_list)):
    im1 = ax1.imshow(im_list[i][0],clim=[-3,3],cmap='bwr')
    # if i == 0:
        # cb = plt.colorbar(im1,ax=cbar_ax)
    im2 = ax2.imshow(im_list[i][1],clim=[-3,3],cmap='bwr')
    im3 = ax3.imshow(im_list[i][2],clim=[-3,3],cmap='bwr')
    ims.append([im1,im2,im3])

ani = anim.ArtistAnimation(fig,ims,interval=33,blit=False)
# plt.show()


writergif = anim.PillowWriter(fps=100)
ani.save('Seq42_Model_Scene_77_Vx.gif', writer=writergif)

# fig = plt.figure(figsize=(20,20))
# ax1=fig.add_subplot(1,3,1)
# ax1.set_title('Ground Truth')
# ax2=fig.add_subplot(1,3,2)
# ax2.set_title('Predicted')
# ax3=fig.add_subplot(1,3,3)
# ax3.set_title('Difference')

# im1 = ax1.imshow(im_list[0][0],clim=[-3,3])
# im2 = ax2.imshow(im_list[0][1],clim=[-3,3])
# im3 = ax3.imshow(im_list[0][2],clim=[-3,3])