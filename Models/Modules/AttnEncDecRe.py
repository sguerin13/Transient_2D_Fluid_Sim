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

# num_epochs = 100
# lr = 1e-3

# define the autoencoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        ########################### 
        #  Convolutional Encoder  #
        ###########################

        # first block - 60 x 120
        self.conv11  = nn.Conv2d(4,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv11.weight)
        self.bn11   = nn.BatchNorm2d(128)
        self.conv12  = nn.Conv2d(128,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv12.weight)
        self.bn12    = nn.BatchNorm2d(128)
        # downsample
        self.conv13  = nn.Conv2d(128,64,2,stride = 2)
        torch.nn.init.xavier_uniform(self.conv13.weight)
        self.bn13    = nn.BatchNorm2d(64)

        # second block - 30 x 60
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

        # third block - 15 x 30
        self.conv31  = nn.Conv2d(32,32,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv31.weight)
        self.bn31    = nn.BatchNorm2d(32)
        self.conv32  = nn.Conv2d(32,32,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv32.weight)
        self.bn32    = nn.BatchNorm2d(32)
        # downsample
        self.conv33  = nn.Conv2d(32,16,3,stride = 3)
        torch.nn.init.xavier_uniform(self.conv33.weight)
        self.bn33   = nn.BatchNorm2d(16)

        # now at 5 x 10 x 16 channels

        ########################### 
        #    Sim Re Encoder    #
        ###########################
        self.Re_encoder = nn.Sequential(
            nn.Linear(1,120*60,bias = True),
            nn.ReLU()
            )

        self.Re_encoder.apply(self.weights_init)
    
    # weight initialiation for the small sub routine
    def weights_init(self,m):
        if type(m) in [nn.Linear]:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.0)


    def forward(self,maps,Re):
        
        # embed and concatenate the reynolds #
        Re_embed   = self.Re_encoder(Re)
        Re_embed   = Re_embed.view(Re_embed.shape[0],1,60,120)
        maps       = torch.cat((maps,Re_embed),axis=1)

        x          = F.relu(self.bn11(self.conv11(maps)))
        x          = F.relu(self.bn12(self.conv12(x)))
        x          = F.relu(self.bn13(self.conv13(x)))
        
        x          = F.relu(self.bn21(self.conv21(x)))
        x          = F.relu(self.bn22(self.conv22(x)))
        x          = F.relu(self.bn23(self.conv23(x)))
        
        x          = F.relu(self.bn31(self.conv31(x)))
        x          = F.relu(self.bn32(self.conv32(x)))
        x          = F.relu(self.bn33(self.conv33(x)))

        return x.view(maps.shape[0],-1)

class Decoder(nn.Module):
    def __init__(self,ch_in):
        super(Decoder,self).__init__()
        self.ch_in = ch_in
        ########################### 
        #  Convolutional Decoder  #
        ###########################
        
        # 1 block - 1D to 15 x 30
        self.dconv11  = nn.ConvTranspose2d(self.ch_in,16,kernel_size = (15,30))
        torch.nn.init.xavier_uniform(self.dconv11.weight)
        self.bn11    = nn.BatchNorm2d(16)
        self.dconv12  = nn.Conv2d(16,32,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv12.weight)
        self.bn12    = nn.BatchNorm2d(32)
        self.dconv13  = nn.ConvTranspose2d(32,64,2,stride = 2)
        torch.nn.init.xavier_uniform(self.dconv13.weight)
        self.bn13    = nn.BatchNorm2d(64)
        
        # 4th block - 32 x 64
        self.dconv21  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv21.weight)
        self.bn21    = nn.BatchNorm2d(64)
        self.dconv22  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv22.weight)
        self.bn22    = nn.BatchNorm2d(64)
        self.dconv23  = nn.ConvTranspose2d(64,128,2,stride = 2)
        torch.nn.init.xavier_uniform(self.dconv23.weight)
        self.bn23    = nn.BatchNorm2d(128)

        # 5th block 64 x 128
        self.dconv31  = nn.Conv2d(128,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv31.weight)
        self.bn31    = nn.BatchNorm2d(128)
        self.dconv32  = nn.Conv2d(128,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv32.weight)
        self.bn32    = nn.BatchNorm2d(128)
        self.dconv33  = nn.Conv2d(128,2,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv33.weight)
 
        
    def forward(self,x):

        x = F.relu(self.bn11(self.dconv11(x)))
        x = F.relu(self.bn12(self.dconv12(x)))
        x = F.relu(self.bn13(self.dconv13(x)))

        x = F.relu(self.bn21(self.dconv21(x)))
        x = F.relu(self.bn22(self.dconv22(x)))
        x = F.relu(self.bn23(self.dconv23(x)))
        
        x = F.relu(self.bn31(self.dconv31(x)))
        x = F.relu(self.bn32(self.dconv32(x)))
        x = self.dconv33(x)

        return x

# class EncoderDecoder(nn.Module):
#     def __init__(self):
#         super(EncoderDecoder,self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.encoder = Encoder()

#         self.decoder = Decoder()

#     def forward(self,map,Re):

#         # x,ind_1,ind_2 = self.encoder(map,params)
#         x = self.encoder(map,Re)
#         # out = self.decoder(x,ind_1,ind_2)
#         out = self.decoder(x)
#         return out

#     def pass_data(self):

#         x = torch.rand(1,3,60,120)
#         Re = torch.rand(1,1)
#         y = self.forward(x,Re)

#         return y

#     def count_parameters(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)
