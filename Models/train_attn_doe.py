'''
script for training attention models while doing parameter search


'''

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.preprocessing import StandardScaler as STD
import Utils.data_utils as du
from Modules.ConvAttnModel import ConvAttn
from Modules.AttnEncDecRe import Encoder,Decoder
import matplotlib.pyplot as plt
import Utils.data_prep as dprep
import os
from model_trainer_multistep_loss_doe import TrainerMulti_StepDOE
from Modules.mask_loss import MaskedLoss
from Modules.BigDataSetR2 import XYDataset
import time
import datetime

torch.cuda.empty_cache()
# cudnn.benchmark = True
# load scalers
direct = os.getcwd()
scaler_file_path = direct + '/../../Data/data/100sims/'
# dprep.build_scalers(path)
velo_scaler = du.load_std_scaler(scaler_file_path+'/scalers/velo_scaler_100.pkl')
rho_scaler  = du.load_std_scaler(scaler_file_path+'/scalers/rho_scaler_100.pkl')
sdf_scaler  = du.load_std_scaler(scaler_file_path+'./scalers/sdf_scaler_100.pkl')
Re_scaler   = du.load_std_scaler(scaler_file_path+'./scalers/Re_scaler_100.pkl')
scaler_list = [velo_scaler,rho_scaler,sdf_scaler,Re_scaler]

# experiment name
exp_name = 'attn_doe_2_5_31'
if not os.path.isdir(os.getcwd()+'/tune/'+exp_name):
    os.mkdir(os.getcwd()+'/tune/'+exp_name)
    
def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))

for i in range(0,100):
    torch.cuda.empty_cache()
    print('run: ',i)
    st_time = time.time()
    # determin sequence length and get the file path
    seq_pairs = [(2,2),(2,3),(4,2),(4,3)]
    pair = np.random.randint(0,4)
    seq_x = seq_pairs[pair][0]
    seq_y = seq_pairs[pair][1]
    data_file_path   = direct + '/../../Data/data/100sims/seqxseqy/'+ str(seq_x) +'x' + str(seq_y) +'y/'

    # get the batch sizes:
    batch_size = np.random.randint(4,9)*4
    
    # let the learning rate vary:
    lr = loguniform(-9.21, -6.9, size=1)[0]

    print(seq_x,seq_y,batch_size)
    # number of attention heads
    if seq_x < 4:
        n_heads= seq_x
    else:
        n_heads=4
    torch.cuda.empty_cache()
    model = ConvAttn(seq_length = seq_x,n_heads=n_heads)
    model.cuda().float()
    

    # initiate the trainer object
    trainer = TrainerMulti_StepDOE(model=model,
                                   file_path = data_file_path,
                                   scalers = scaler_list)
    
    # set up training params
    trainer.setup_training(loss_criteria=MaskedLoss(nn.MSELoss()),
                           optimizer = torch.optim.Adam,
                           lr = lr,
                           loss_steps = seq_y,
                           batch_size = batch_size,
                           num_updates = 500)

    # create the data loaders
    trainer.build_data_loaders()
    # run for 'num_updates'
    try:
        loss = trainer.train_net()

        hyp_dict = {}
        hyp_dict['loss'] = loss
        hyp_dict['seq_x'] = seq_x
        hyp_dict['seq_y'] = seq_y
        hyp_dict['batch'] = batch_size
        hyp_dict['lr'] = lr
        hyp_dict['n_heads'] = n_heads

        file_str = os.getcwd()+'/tune/'+exp_name+'/run'+str(i)+'.pt'
        torch.save(hyp_dict,file_str)

        end_time = time.time()
        tr_time = end_time-st_time
        print(tr_time)
        print('datetime', datetime.datetime.now())
    except:
        print('training failed, run: ',i)
        print('batch: ',batch_size,', input_len: ',seq_x)


