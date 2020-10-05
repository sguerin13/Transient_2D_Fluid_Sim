'''
script for training models

'''

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from sklearn.preprocessing import StandardScaler as STD

import Utils.data_prep as dprep
import Utils.data_utils as du
from Modules.AttnEncDecRe import Decoder, Encoder
from Modules.ConvAttnModel import ConvAttn
from model_trainer_multi_step import Trainer_MultiStep
# from Modules.BigDataSet import BigAssDataset
from Modules.BigDataSetR2 import XYDataset
from Modules.mask_loss import MaskedLoss
# from Modules.EncoderDecoderSDF import Encoder,Decoder,EncoderDecoder
# from Modules.SimpleEncDecRe import Encoder,Decoder,EncoderDecoder
# from Modules.SeqConvEncoderDecoder import Decoder, Encoder, EncoderDecoder

# from Utils.data_prep import concat_collate

torch.cuda.empty_cache()
cudnn.benchmark = False
# load scalers
direct = os.getcwd()
scaler_file_path = direct + '/../../Data/data/100sims/'
data_file_path   = direct + '/../../Data/data/100sims/seqxseqy/4x2y/'

# dprep.build_scalers(path)
velo_scaler = du.load_std_scaler(scaler_file_path+'/scalers/velo_scaler_100.pkl')
rho_scaler  = du.load_std_scaler(scaler_file_path+'/scalers/rho_scaler_100.pkl')
sdf_scaler  = du.load_std_scaler(scaler_file_path+'./scalers/sdf_scaler_100.pkl')
Re_scaler   = du.load_std_scaler(scaler_file_path+'./scalers/Re_scaler_100.pkl')
scaler_list = [velo_scaler,rho_scaler,sdf_scaler,Re_scaler]

# load model
model = ConvAttn(seq_length = 4,n_heads=4,model_dropout=0,device='gpu')
model.cuda().float()
model_file_name = 'ATTN_4in_2out_Run2_6_4.pth'

# initiate the trainer object
trainer = Trainer_MultiStep(model=model,
                  scalers = scaler_list,
                  model_file_name = model_file_name,
                  file_path = data_file_path,
                  board_file = 'ATTN_4in_2out_Run2_6_4')
# trainer.device = torch.device('cpu')
# set up training params
trainer.setup_training(val_split=.1,
                      loss_criteria=MaskedLoss(nn.MSELoss()),
                      optimizer = torch.optim.Adam,
                      epochs = 25,
                      min_eps = 5,
                      lr = 3e-4,
                      batch_size = 20)

# create the data loaders
trainer.build_data_loaders()
trainer.train_net(save_model=True)
