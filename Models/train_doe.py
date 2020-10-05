'''
Script for training models while doing parameter search


'''



import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from sklearn.preprocessing import StandardScaler as STD
import Utils.data_utils as du
# from Modules.EncoderDecoderSDF import Encoder,Decoder,EncoderDecoder
# from Modules.SeqConvEncoderDecoder import Encoder,Decoder,EncoderDecoder
from Modules.ConvAttnModel import ConvAttn
from Modules.AttnEncDecRe import Encoder,Decoder
import matplotlib.pyplot as plt
import Utils.data_prep as dprep
import os
from model_trainer_doe import TrainerXYDOE
from Modules.mask_loss import MaskedLoss
# from Modules.BigDataSet import BigAssDataset
from Modules.BigDataSetR2 import XYDataset
# from Utils.data_prep import concat_collate

torch.cuda.empty_cache()
cudnn.benchmark = True
# load scalers
direct = os.getcwd()
scaler_file_path = direct + '/../../Data/data/100sims/'
# dprep.build_scalers(path)
velo_scaler = du.load_std_scaler(scaler_file_path+'/scalers/velo_scaler_100.pkl')
rho_scaler  = du.load_std_scaler(scaler_file_path+'/scalers/rho_scaler_100.pkl')
sdf_scaler  = du.load_std_scaler(scaler_file_path+'./scalers/sdf_scaler_100.pkl')
Re_scaler   = du.load_std_scaler(scaler_file_path+'./scalers/Re_scaler_100.pkl')
scaler_list = [velo_scaler,rho_scaler,sdf_scaler,Re_scaler]

sub_epochs = .5
batch_size = 16

for i in range(8,11):

    data_file_path   = direct + '/../../Data/data/100sims/seqxy/x' + str(i) +'y/'
    # load model
    # model = EncoderDecoder(num_scenes = i).double()
    if i < 4:
        n_heads=i
    else:
        n_heads=4

    model = ConvAttn(seq_length = i,n_heads=n_heads,device = 'cpu').double()
    model.cuda().float()
    model_file_name = 'ATTN_'+ str(i) + '_100_sims_quarter_epoch_doe.pth'

    # initiate the trainer object
    trainer = TrainerXYDOE(model=model,
                    scalers = scaler_list,
                    model_file_name = model_file_name,
                    file_path = data_file_path,
                    board_file = 'ATTN_'+ str(i) + '_100_sims_quarter_epoch_doe')
    torch.cuda.empty_cache()
    
    # set up training params
    trainer.setup_training(val_split=None,
                           loss_criteria=MaskedLoss(nn.MSELoss()),
                           optimizer = torch.optim.Adam,
                           epochs = 1,
                           lr = 1e-4,
                           batch_size = batch_size,
                           sub_epochs = sub_epochs)

    # create the data loaders
    trainer.build_data_loaders()
    trainer.train_net(save_model=True)

