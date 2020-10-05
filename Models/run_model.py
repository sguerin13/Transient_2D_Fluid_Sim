'''
method to load and run the model

'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler,DataLoader,TensorDataset,random_split
from SimpleEncDecRe import Encoder,Decoder,EncoderDecoder
from Modules.mask_loss import MaskedLoss
from Modules.BigDataSet import BigAssDataset
from Modules.BigDataSetR2 import XYDataset
from Utils.data_prep import concat_collate
import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
state_dict_path = os.getcwd()+'/saved_models/ConvEncDecRe_100_sims_10_epochs_test/ConvEncDecRe_100_sims_10_epochs_test.pth'
data_file_path   = os.getcwd() + '/../../Data/data/100sims/xy/'

# load model
torch.cuda.empty_cache()
torch.set_grad_enabled(False)
model = EncoderDecoder().cuda().float()
model.load_state_dict(torch.load(state_dict_path))
model.float()
model.eval()

# create dataset

# load file paths
files = os.listdir(data_file_path)
dataset = XYDataset(path=data_file_path,scalers=None)
dloader = DataLoader(dataset,shuffle=True,batch_size=1)


# may need to load scalers and unscale
def load_sample(data,loss_steps): 
    # send data to gpu
    x    = data[0].cuda().float()
    x    = x.unsqueeze(0)
    Re   = data[1].cuda().float()
    Re   = Re.unsqueeze(0)
    mask = data[2].cuda().float()
    y    = data[3].cuda().float()

    mask = torch.unsqueeze(mask,dim=1)
    return x,Re,mask,y

def pass_through_model(model,x,Re):
    out = model.forward(x,Re)
    out_numpy = out.cpu().detach().numpy()
    return out, out_numpy

def vorticity(x,y):
    vort = np.abs(x[1:-1, 2:] - x[1:-1, :-2]
                - y[2:, 1:-1] + y[:-2, 1:-1])
    return vort

def compare_results(out,y,mask,metric='vx',iterr=0,plot=True):
    out_masked = out*mask.cpu().detach().numpy()
    out_x = out_masked[0,0,:,:]
    out_y = out_masked[0,1,:,:]
    y_masked   = y.cpu().detach().numpy()*mask.cpu().detach().numpy()
    y_x = y_masked[0,0,:,:]
    y_y = y_masked[0,1,:,:]

    if plot==True:
        if iterr%10 == 0:
                
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

def main(loss_steps = 1,metric = 'vx',mode='save',scene_to_save = 1,plot=False):
    av_loss_matrix = np.zeros((100,240))
    for i in range(100):
        torch.cuda.empty_cache()
        print(i)

        for j in range(201):
            file_str = '100sims_sim_'+str(i)+'sample_'+str(j)+'.pt'
            data = torch.load(data_file_path + file_str)
            
            if j==0:
                x,Re,mask,y = load_sample(data,loss_steps=1)
                out,out_numpy = pass_through_model(model,x,Re)
                av_loss_matrix[i,j] = compare_results(out_numpy,y,
                                                    mask,metric=metric,
                                                    iterr=j,plot=plot)
                # update the input with the output from the model
                x[:,:-1,:,:] = out
                del out,out_numpy
            else:
                _,_,_,y = load_sample(data,loss_steps=1)
                out,out_numpy = pass_through_model(model,x,Re)
                av_loss_matrix[i,j] = compare_results(out_numpy,y,
                                                    mask,metric=metric,
                                                    iterr=j,plot=plot)
                x[:,:-1,:,:] = out
                del out,out_numpy

    return  av_loss_matrix

av_loss_matrix = main(metric = 'vy',mode='save',plot=False,scene_to_save = 1)
av_loss_over_time = [np.mean(av_loss_matrix[:,i]) for i in range(av_loss_matrix.shape[1])]
save_file_path = state_dict_path = os.getcwd() + '/model_comparison/SISO.pt'
torch.save(av_loss_over_time,save_file_path)


