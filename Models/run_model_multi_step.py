'''
method to load and run the model with a model that uses multiple steps as
inputs

'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler,DataLoader,TensorDataset,random_split
# from SeqConvEncoderDecoder import Decoder, Encoder, EncoderDecoder
from ConvAttnModel import ConvAttn
from Modules.mask_loss import MaskedLoss
from Modules.BigDataSet import BigAssDataset
from Modules.BigDataSetR2 import XYDataset
from Utils.data_prep import concat_collate
import tqdm
import time
from torch.utils.tensorboard import SummaryWriter


state_dict_path = os.getcwd()+'/saved_models/ATTN_4in_2out_Run2_6_4/ATTN_4in_2out_Run2_6_4.pth'
data_file_path   = os.getcwd() + '/../../Data/data/100sims/seqxy/x4y/'

# model = EncoderDecoder(num_scenes = 4).float()
model = ConvAttn(seq_length = 4,n_heads=4,device = 'gpu').float()
model.load_state_dict(torch.load(state_dict_path))
model.cuda().float()

torch.set_grad_enabled(False)
model.eval()

# load file paths
files = os.listdir(data_file_path)
# may need to load scalers and unscale

def load_sample(data,loss_steps): 
    # send data to gpu
    x    = data[0].cuda().float()
    x = x.unsqueeze(0)
    Re   = data[1].cuda().float()
    Re = Re.unsqueeze(0)
    mask = data[2].cuda().float()
    y    = data[3].cuda().float()
    # print(x.shape,Re.shape,mask.shape,y.shape)
    mask = torch.unsqueeze(mask,dim=1)
    if loss_steps == 2:
        mask = torch.cat((mask,mask),dim=1)
    elif loss_steps == 3:
        mask = torch.cat((mask,mask,mask),dim==1)
    else:
        pass
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

def compare_results(out,y,mask,metric='vx',iterr=0,plot=True):
    out_masked = out*mask.cpu().detach().numpy()
    out_x = out_masked[0,0,:,:]
    out_y = out_masked[0,1,:,:]
    y_masked   = y.detach().cpu().numpy()*mask.cpu().detach().numpy()
    # print(y_masked.shape)
    y_x = y_masked[0,0,:,:]
    y_y = y_masked[0,1,:,:]
    if plot==True:
        if metric == 'vx':
            plt.figure(figsize=(10,10))
            plt.suptitle('X velocity'+ ' time step '+str(iterr),y=.62)
            plt.subplot(1,3,1)
            plt.imshow(y_x,clim=[-3,3])
            plt.title('Target')
            plt.subplot(1,3,2)
            plt.imshow(out_x,clim=[-3,3])
            plt.title('Prediction')
            plt.subplot(1,3,3)
            plt.imshow(y_x-out_x,clim=[-3,3])
            # print(np.mean(np.abs(y_x-out_x)))
            plt.title('Delta')
            plt.show()

        if metric == 'vy':
            plt.figure(figsize=(10,10))
            plt.suptitle('Y velocity'+' time step '+str(iterr),y=.62)
            plt.subplot(1,3,1)
            plt.imshow(y_y,clim=[-3,3])
            plt.title('Target')
            plt.subplot(1,3,2)
            plt.imshow(out_y,clim=[-3,3])
            plt.title('Prediction')
            plt.subplot(1,3,3)
            plt.imshow(y_y-out_y,clim=[-3,3])
            # print(np.mean(np.abs(y_y-out_y)))
            plt.title('Delta')
            plt.show()

        if metric == 'vorticity':
            gt_vort = vorticity(y_x,y_y)
            pred_vort = vorticity(out_x,out_y)
            plt.figure(figsize=(10,10))
            plt.suptitle('Y velocity'+' time step '+str(iterr),y=.62)
            plt.subplot(1,3,1)
            plt.imshow(gt_vort,clim=[-3,3])
            plt.title('Target')
            plt.subplot(1,3,2)
            plt.imshow(pred_vort,clim=[-3,3])
            plt.title('Prediction')
            plt.subplot(1,3,3)
            plt.imshow(gt_vort - pred_vort,clim=[-3,3])
            # print(np.mean(np.abs(gt_vort - pred_vort)))
            plt.title('Delta')
            plt.show()
    
    av_x_loss = np.mean(np.abs(out_x - y_x))
    av_y_loss = np.mean(np.abs(out_y - y_y))
    av_loss   = (av_x_loss + av_y_loss)/2
    return av_loss


def main(loss_steps = 1,multi_in = True,metric = 'vx',mode='save',scene_to_save = 1,plot=False):
    av_loss_matrix = np.zeros((100,240))
    for i in range(100):
        print(i)

        for j in range(240):
            file_str = '100sims_sim_'+str(i)+'sample_'+str(j)+'.pt'
            data = torch.load(data_file_path + file_str)

            if j == 0:
                if multi_in == True:
                    x,Re,mask,y = load_sample(data,loss_steps=1)
                    out,out_numpy = pass_through_model(model,x,Re)
                    # print(out.shape)
                    av_loss_matrix[i,j] = compare_results(out_numpy,y,
                                                        mask,metric=metric,
                                                        iterr=j,plot=False)
                    # update the input with the output from the model
                    x[:,-1,:-1,:,:] = out
                    del out,out_numpy
                else:
                    x,Re,mask,y = load_sample(data,loss_steps=1)
                    out,out_numpy = pass_through_model(model,x,Re)
                    av_loss_matrix[i,j] = compare_results(out_numpy,y,
                                                        mask,metric=metric,
                                                        iterr=j,plot=False)
                    # update the input with the output from the model
                    x[:,:-1,:,:] = out
                    del out,out_numpy
            
            else:
                if multi_in == True:
                    _,_,_,y = load_sample(data,loss_steps=1)
                    out,out_numpy = pass_through_model(model,x,Re)
                    av_loss_matrix[i,j] = compare_results(out_numpy,y,
                                                        mask,metric=metric,
                                                        iterr=j,plot=False)
                    # update the input with the output from the model
                    x[:,-1,:-1,:,:] = out
                    del out,out_numpy
                else:
                    _,_,_,y = load_sample(data,loss_steps=1)
                    out,out_numpy = pass_through_model(model,x,Re)
                    av_loss_matrix[i,j] = compare_results(out_numpy,y,
                                                        mask,metric=metric,
                                                        iterr=j,plot=False)
                    x[:,:-1,:,:] = out
                    del out,out_numpy
            
    return  av_loss_matrix

av_loss_matrix = main(loss_steps = 1,multi_in = True,metric = 'vorticity',mode='save',scene_to_save = 1)


av_loss_over_time = [np.mean(av_loss_matrix[:,i]) for i in range(av_loss_matrix.shape[1])]
save_file_path = state_dict_path = os.getcwd() + '/model_comparison/ATTN_4_2.pt'
torch.save(av_loss_over_time,save_file_path)