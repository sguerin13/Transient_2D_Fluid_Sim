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
import copy

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
        ''' 
        
        '''

        # first block - 64 x 128
        self.conv11  = nn.Conv2d(4,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv11.weight)
        self.bn11   = nn.BatchNorm2d(64)
        self.conv12  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv12.weight)
        self.bn12    = nn.BatchNorm2d(64)
        # downsample
        self.conv13  = nn.Conv2d(64,64,3,stride = 2,padding=1)
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
        self.conv23  = nn.Conv2d(64,64,3,stride = 2,padding=1)
        torch.nn.init.xavier_uniform(self.conv23.weight)
        self.bn23    = nn.BatchNorm2d(64)

        # third block - 16 x 32
        self.conv31  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv31.weight)
        self.bn31    = nn.BatchNorm2d(64)
        self.conv32  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv32.weight)
        self.bn32    = nn.BatchNorm2d(64)
        # downsample
        self.conv33  = nn.Conv2d(64,64,3,stride = 2,padding=1)
        torch.nn.init.xavier_uniform(self.conv33.weight)
        self.bn33    = nn.BatchNorm2d(64)

        # 4th block - 8 x 16
        self.conv41  = nn.Conv2d(64,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv41.weight)
        self.bn41    = nn.BatchNorm2d(128)
        self.conv42  = nn.Conv2d(128,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv42.weight)
        self.bn42    = nn.BatchNorm2d(128)
        # downsample
        self.conv43  = nn.Conv2d(128,128,3,stride = 2,padding=1)
        torch.nn.init.xavier_uniform(self.conv43.weight)
        self.bn43    = nn.BatchNorm2d(128)

        # 5th block - 4 x 8
        self.conv51  = nn.Conv2d(128,256,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv51.weight)
        self.bn51    = nn.BatchNorm2d(256)
        self.conv52  = nn.Conv2d(256,256,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.conv52.weight)
        self.bn52    = nn.BatchNorm2d(256)
        self.conv53  = nn.Conv2d(256,256,3,stride = 2,padding=1)
        torch.nn.init.xavier_uniform(self.conv53.weight)
        self.bn53    = nn.BatchNorm2d(256)
        # now at 2 x 4 x 256 = 2048

        ## ** Can experiment w/ flattening this into a series of 1D convolutions

        ########################### 
        #    Sim Param Encoder    #
        ###########################
        self.param_encoder = nn.Sequential(
            nn.Linear(3,1024,bias = True),
            nn.ReLU(),
            nn.Linear(1024,2048,bias=True),
            nn.ReLU())

        self.param_encoder.apply(self.weights_init)
    
    # weight initialiation for the small sub routine
    def weights_init(self,m):
        if type(m) in [nn.Linear]:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.0)



    def forward(self,maps,params):

        x          = F.relu(self.bn11(self.conv11(maps)))
        x          = (F.relu(self.bn12(self.conv12(x))))
        x          = (F.relu(self.bn13(self.conv13(x))))
        
        x          = F.relu(self.bn21(self.conv21(x)))
        x          = F.relu(self.bn22(self.conv22(x)))
        x          = F.relu(self.bn23(self.conv23(x)))
        
        x          = F.relu(self.bn31(self.conv31(x)))
        x          = F.relu(self.bn32(self.conv32(x)))
        x          = F.relu(self.bn33(self.conv33(x)))
        
        x          = F.relu(self.bn41(self.conv41(x)))
        x          = F.relu(self.bn42(self.conv42(x)))
        x          = F.relu(self.bn43(self.conv43(x)))
        
        x          = F.relu(self.bn51(self.conv51(x)))
        x          = F.relu(self.bn52(self.conv52(x)))
        x          = F.relu(self.bn53(self.conv53(x)))


        # x_conv = x.view(x.shape[0],-1) # account for batch size
        x_param = self.param_encoder(params)
        # x_out = x_conv+x_param
        # x_out = torch.cat((x_conv,x_param),dim=1)
        x_out = x

        return x_out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ########################### 
        #  Convolutional Decoder  #
        ###########################
        # o = (i-1)*s + k -2p

        # 1st block - 4 x 8
        self.dconv11  = nn.ConvTranspose2d(256,256,2,stride = 2)
        torch.nn.init.xavier_uniform(self.dconv11.weight)
        self.bn11    = nn.BatchNorm2d(256)
        self.dconv12  = nn.Conv2d(256,256,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv12.weight)
        self.bn12    = nn.BatchNorm2d(256)
        self.dconv13  = nn.Conv2d(256,256,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv13.weight)
        self.bn13    = nn.BatchNorm2d(256)
        

        # 2th block - 8 x 16
        self.dconv21  = nn.ConvTranspose2d(256,128,2,stride = 2)
        torch.nn.init.xavier_uniform(self.dconv21.weight)
        self.bn21    = nn.BatchNorm2d(128)
        self.dconv22  = nn.Conv2d(128,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv22.weight)
        self.bn22    = nn.BatchNorm2d(128)
        self.dconv23  = nn.Conv2d(128,128,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv23.weight)
        self.bn23    = nn.BatchNorm2d(128)
        

        # 3th block - 16 x 32
        self.dconv31  = nn.ConvTranspose2d(128,64,2,stride = 2)
        torch.nn.init.xavier_uniform(self.dconv31.weight)
        self.bn31    = nn.BatchNorm2d(64)
        self.dconv32  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv32.weight)
        self.bn32    = nn.BatchNorm2d(64)
        self.dconv33  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv33.weight)
        self.bn33   = nn.BatchNorm2d(64)
        

        # 4th block - 32 x 64
        self.dconv41  = nn.ConvTranspose2d(64,64,2,stride = 2)
        torch.nn.init.xavier_uniform(self.dconv41.weight)
        self.bn41    = nn.BatchNorm2d(64)
        self.dconv42  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv42.weight)
        self.bn42    = nn.BatchNorm2d(64)
        self.dconv43  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv43.weight)
        self.bn43    = nn.BatchNorm2d(64)

        # 5th block - 64 x 128
        self.dconv51  = nn.ConvTranspose2d(64,64,2,stride = 2)
        torch.nn.init.xavier_uniform(self.dconv51.weight)
        self.bn51    = nn.BatchNorm2d(64)
        self.dconv52  = nn.Conv2d(64,64,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv52.weight)
        self.bn52    = nn.BatchNorm2d(64) 
        self.dconv53  = nn.Conv2d(64,3,3,stride = 1,padding=1)
        torch.nn.init.xavier_uniform(self.dconv53.weight)
        self.bn53    = nn.BatchNorm2d(3) # probably not needed
        

    def forward(self,x):#,x,max_ind_1,max_ind_2):

        # x = x.view(x.shape[0],512,2,4) # give it the same shape as it was before we flattened it
        
        x = F.relu(self.bn11(self.dconv11(x)))
        x = F.relu(self.bn12(self.dconv12(x)))
        x = F.relu(self.bn13(self.dconv13(x)))
        
        x = F.relu(self.bn21(self.dconv21(x)))
        x = F.relu(self.bn22(self.dconv22(x)))
        x = F.relu(self.bn23(self.dconv23(x)))

        x = F.relu(self.bn31(self.dconv31(x)))
        x = F.relu(self.bn32(self.dconv32(x)))
        x = F.relu(self.bn33(self.dconv33(x)))

        x = F.relu(self.bn41(self.dconv41(x)))
        x = F.relu(self.bn42(self.dconv42(x)))
        x = F.relu(self.bn43(self.dconv43(x)))

        x = F.relu(self.bn51(self.dconv51(x)))
        x = F.relu(self.bn52(self.dconv52(x)))

        y = self.dconv53(x)
        
        return y

class EncoderDecoder(nn.Module):
    def __init__(self,Enc,Dec):
        super(EncoderDecoder,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Enc
        # self.encoder.to(self.device)
        self.decoder = Dec
        # self.decoder.to(self.device)

    def forward(self,map,params):

        # x,ind_1,ind_2 = self.encoder(map,params)
        x = self.encoder(map,params)
        # out = self.decoder(x,ind_1,ind_2)
        out = self.decoder(x)
        return out

    def train_net(self,dataset,lr = 1e-3,epochs = 100):

        criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr = lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',factor=.2,patience =50)
        best_loss = 10
        self.loss_vec = []
        for eps in range(epochs):
            for idx,(data) in enumerate(dataset): # data_loader inputs are (x,param_vec,targets)
                
                x = data[0].to(self.device)
                p = data[1].to(self.device)
                t = data[2].to(self.device)
                # forward pass
                output = self.forward(x,p)
                loss = criterion(output,t)
                self.loss_vec.extend([loss.cpu().detach().numpy()])
                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if loss < best_loss:
                    self.best_model_state = self.state_dict()
                
            self.scheduler.step(loss)
            print('epoch %i / %i, loss: %f, LR:',(eps+1,epochs,loss.data,self.scheduler._last_lr))

    def pass_data(self):

        x = torch.rand(1,4,64,128)
        params = torch.rand(1,3)
        y = self.forward(x,params)

        return y


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
