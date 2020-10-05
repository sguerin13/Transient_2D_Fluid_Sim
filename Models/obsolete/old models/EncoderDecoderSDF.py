'''
Convolutional Encoder/Decoder Architecture

- Takes 2D Velocity and density maps and SDF and jointly embeds with the convolution
- Takes Input Parameters and Embeds them into a FC layer the same dimension as the FC Layer

'''

#### Pytorch autoencoder ####

import torch
from torch.autograd import Variable
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

        # Convolutional Encoder
        self.conv1  = nn.Conv2d(4,32,12,stride = 4) # y_dim = 64, x_dim = 128
        self.pool1  = nn.AvgPool2d(kernel_size = 4,stride = 2)#,return_indices = True) # y_size,x_size = (14,30)
        self.conv2  = nn.Conv2d(32,64,kernel_size = (2,4),stride = (1,2)) # y_size,x_size = (6,14)
        self.pool2  = nn.AvgPool2d(kernel_size = 3,stride=1)#,return_indices = True) # y_size,x_size = (5,6)
        self.conv3  = nn.Conv2d(64,96,kernel_size = 2) # y_size,x_size = (3,4)
        # output is of shape 96*2*3 = 576

        # Param Encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(3,288,bias = True),
            nn.ReLU(),
            nn.Linear(288,576,bias=True)
        )
    
    def forward(self,maps,params):
        # x,max_ind_1 = self.pool1(F.relu(self.conv1(maps)))
        # x,max_ind_2 = self.pool2(F.relu(self.conv2(x)))
        x = self.pool1((self.conv1(maps)))
        x = self.pool2((self.conv2(x)))
        x = self.conv3((x))
        x_flat = x.view(x.shape[0],-1) # account for batch size
        # x_2 = self.param_encoder(params)
        # x_out = x_flat+x_2
        #x_out = torch.cat((x_flat,x_2),dim=1)
        x_out = x_flat
        
        return x_out #,max_ind_1,max_ind_2


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        #decoder
        
        # O = s*(I-1)+K - 2P
        self.deconv1 = nn.ConvTranspose2d(96,64,kernel_size = 2,stride=2) # 96 x 2 x 3
        self.deconv2 = nn.ConvTranspose2d(64,48,kernel_size = 4,stride=2) # 64 x 4x6
        self.deconv3 = nn.ConvTranspose2d(48,32,kernel_size = 8,stride=(2,4)) # 48 x 10 x 14
        self.deconv4 = nn.ConvTranspose2d(32,3,kernel_size = 16,stride=(2),padding = (1,3)) # 40 x 26 x 60
        
        # # self.unpool1 = nn.MaxUnpool2d(kernel_size=3,stride = 1)
        # self.deconv2 = nn.ConvTranspose2d(64,32,kernel_size = (2,4),stride=(1,2))
        # # self.unpool2 = nn.MaxUnpool2d(kernel_size=4,stride=2)
        # self.deconv3 = nn.ConvTranspose2d(32,3,kernel_size =12,stride = 4)

    def forward(self,x):#,x,max_ind_1,max_ind_2):
        x = x.view(x.shape[0],96,2,3) # give it the same shape as it was before we flattened it
        # x = self.unpool1(F.relu(self.deconv1(x)),max_ind_2)
        # x = self.unpool2(F.relu(self.deconv2(x)),max_ind_1)

        y = self.deconv4(self.deconv3(self.deconv2(self.deconv1(x)))) 
        return y

class EncoderDecoder(nn.Module):
    def __init__(self,Enc,Dec):
        super(EncoderDecoder,self).__init__()
        
        self.encoder = Enc
        self.decoder = Dec

    def forward(self,map,params):

        # x,ind_1,ind_2 = self.encoder(map,params)
        x = self.encoder(map,params)
        # out = self.decoder(x,ind_1,ind_2)
        out = self.decoder(x)
        return out

    def train(self,dataset,lr = 1e-3,epochs = 100):

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(),lr = lr)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=.5,patience =5)

        for eps in range(epochs):

            for idx,(data) in enumerate(dataset): # data_loader inputs are (x,param_vec,targets)
                
                x = data[0]
                p = data[1]
                t = data[2]
                # forward pass
                output = self.forward(x,p)
                loss = criterion(output,t)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #scheduler.step(loss)
            print('epoch %i / %i, loss: %f',(eps+1,epochs,loss.data))

