'''
Convolutional Encoder/Decoder Architecture

- Takes 2D Velocity and density maps and SDF and jointly embeds with the convolution
- Takes Input Parameters and Embeds them into a FC layer the same dimension as the FC Layer

'''

#### Pytorch autoencoder ####

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F 
from Utils.data_utils import sim_to_auto_enc_data
from Utils.data_utils import np_to_torch_dataloader
import numpy as np

# # load data and create data loader
# x,y = sim_to_auto_enc_data('data/Re_25.npz')
# print(x.shape)
# # ignore normalization for the time being
# data_loader = np_to_torch_dataloader(x,y,batch = 50)

# num_epochs = 100
# lr = 1e-3

# define the autoencoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ########################### 
        #  Convolutional Encoder  #
        ###########################


        # first block - 64 x 128
        self.conv11  = nn.Conv2d(3,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv11.weight)
        self.bn11   = nn.BatchNorm2d(128)
        self.conv12  = nn.Conv2d(128,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv12.weight)
        self.bn12    = nn.BatchNorm2d(128)
        # downsample
        self.conv13  = nn.Conv2d(128,64,2,stride = 2)
        torch.nn.init.xavier_uniform(self.conv13.weight)
        self.bn13    = nn.BatchNorm2d(64)

        # second block - 32 x 64
        self.conv21  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv21.weight)
        self.bn21    = nn.BatchNorm2d(64)
        self.conv22  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv22.weight)
        self.bn22   = nn.BatchNorm2d(64)
        # downsample
        self.conv23  = nn.Conv2d(64,32,2,stride = 2)
        torch.nn.init.xavier_uniform(self.conv23.weight)
        self.bn23   = nn.BatchNorm2d(32)

        # third block - 16 x 32
        self.conv31  = nn.Conv2d(32,32,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv31.weight)
        self.bn31    = nn.BatchNorm2d(32)
        self.conv32  = nn.Conv2d(32,32,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv32.weight)
        self.bn32    = nn.BatchNorm2d(32)


    def forward(self,maps,Re):

        x          = F.relu(self.bn11(self.conv11(maps)))
        x          = F.relu(self.bn12(self.conv12(x)))
        x          = F.relu(self.bn13(self.conv13(x)))
        
        x          = F.relu(self.bn21(self.conv21(x)))
        x          = F.relu(self.bn22(self.conv22(x)))
        x          = F.relu(self.bn23(self.conv23(x)))
        
        x          = F.relu(self.bn31(self.conv31(x)))
        x          = F.relu(self.bn32(self.conv32(x)))

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ########################### 
        #  Convolutional Decoder  #
        ###########################
        
        # 3th block - 16 x 32
        self.dconv31  = nn.Conv2d(32,32,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv31.weight)
        self.bn31    = nn.BatchNorm2d(32)
        self.dconv32  = nn.Conv2d(32,32,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv32.weight)
        self.bn32    = nn.BatchNorm2d(32)
        self.dconv33  = nn.ConvTranspose2d(32,64,2,stride = 2)
        torch.nn.init.xavier_uniform(self.dconv33.weight)
        self.bn33    = nn.BatchNorm2d(64)
        
        # 4th block - 32 x 64
        self.dconv41  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv41.weight)
        self.bn41    = nn.BatchNorm2d(64)
        self.dconv42  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv42.weight)
        self.bn42    = nn.BatchNorm2d(64)
        self.dconv43  = nn.ConvTranspose2d(64,128,2,stride = 2)
        torch.nn.init.xavier_uniform(self.dconv43.weight)
        self.bn43    = nn.BatchNorm2d(128)

        # 5th block 64 x 128
        self.dconv51  = nn.Conv2d(128,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv51.weight)
        self.bn51    = nn.BatchNorm2d(128)
        self.dconv52  = nn.Conv2d(128,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv52.weight)
        self.bn52    = nn.BatchNorm2d(128)
        self.dconv53  = nn.Conv2d(128,2,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv53.weight)
 
    def forward(self,x):

        x = F.relu(self.bn31(self.dconv31(x)))
        x = F.relu(self.bn32(self.dconv32(x)))
        x = F.relu(self.bn33(self.dconv33(x)))

        x = F.relu(self.bn41(self.dconv41(x)))
        x = F.relu(self.bn42(self.dconv42(x)))
        x = F.relu(self.bn43(self.dconv43(x)))
        
        x = F.relu(self.bn51(self.dconv51(x)))
        x = F.relu(self.bn52(self.dconv52(x)))
        x = self.dconv53(x)

        return x

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder()

        self.decoder = Decoder()

    def forward(self,map,Re):

        x = self.encoder(map,Re)
        out = self.decoder(x)
        return out

    def pass_data(self):

        x = torch.rand(1,3,60,120)
        params = torch.rand(1,1)
        y = self.forward(x,params)

        return y

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
