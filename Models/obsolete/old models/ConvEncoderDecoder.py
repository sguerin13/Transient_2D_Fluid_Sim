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

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,12,stride = 4), # y_dim = 64, x_dim = 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4,stride = 2) # y_size,x_size = (14,30)
            nn.Conv2d(32,64,kernel_size = (2,4),stride = (1,2), # y_size,x_size = (6,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3) # y_size,x_size = (5,6)
            nn.Conv2d(64,96,kernel_size = 2) # y_size,x_size = (3,4)
            # output is of shape 96*2*3 = 576
        )

        #decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,96,kernel_size = (3,4),stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96,64,kernel_size = (2,4),stride=(1,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size =6,stride=(1,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(32,3,12,stride = 4)
            )

    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


    def train(self,lr = 1e-3,epochs = 100):

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(),lr = lr)

        for eps in range(epochs):

            for data in data_loader:
                x,_ = data
                # print(len(x))
                # print(x[0],x[1])
                x = Variable(x.float())

                # forward pass
                output = self.forward(x)
                loss = criterion(output,x)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('epoch %i / %i, loss: %f',(eps+1,epochs,loss.data))







