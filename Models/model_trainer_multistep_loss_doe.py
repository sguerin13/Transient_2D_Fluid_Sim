'''
Class to handle all of the training tasks parameter search for models that 
use multiple time step losses

Decouples the Training from the nn.Module definition

'''
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import SubsetRandomSampler,DataLoader,TensorDataset,random_split
from Modules.mask_loss import MaskedLoss
from Modules.BigDataSet import BigAssDataset
from Modules.BigDataSetR2 import XYDataset
from Utils.data_prep import concat_collate
import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import os

class TrainerMulti_StepDOE():

    def __init__(self,model,scalers,file_path):
        self.model = model # NN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_path = file_path # file path of the data_set
        self.scalers = scalers
        self.dataset = XYDataset(path=self.file_path,scalers=self.scalers)

    def setup_training(self,loss_criteria,optimizer,lr,
                       batch_size = 32,num_updates = 500, loss_steps = 2):
        self.lr = lr
        self.loss_criteria = loss_criteria # MaskedLoss(nn.MSELoss())
        self.optimizer = optimizer(self.model.parameters(),lr=lr) #torch.optim.Adam(self.model.parameters(),lr = lr)
        self.batch_size = batch_size
        self.num_updates = num_updates
        self.loss_steps = loss_steps

    def build_data_loaders(self):
        self.dloader = DataLoader(self.dataset,batch_size=self.batch_size,
                                    pin_memory=False,drop_last=False,shuffle=True)

    def train_net(self):
        self.loss_vec = []
        updates = 0
        torch.set_grad_enabled(True)
        self.model.train()

        while updates < self.num_updates:

            for idx,(data) in enumerate(self.dloader):               
                # send data to gpu
                x    = data[0].to(self.device).float()
                Re   = data[1].to(self.device).float()
                mask = data[2].to(self.device).float()
                mask = torch.unsqueeze(mask,dim=1)
                if self.loss_steps == 2:
                    mask = torch.cat((mask,mask),dim=1)
                elif self.loss_steps == 3:
                    mask = torch.cat((mask,mask,mask),dim==1)
                else:
                    pass

                y    = data[3].to(self.device).float()
                # forward pass
                output = self.model.forward(x,Re)
                output = torch.unsqueeze(output,1)
                multi_step_output = output #place holder
                # calculate multistep output
                # print('y: ',y.shape)
                
                for _ in range(1,self.loss_steps):
                    x[:,-1,:-1,:,:] = output.view(output.shape[0],output.shape[2],output.shape[3],output.shape[4])    # replace vx, vy of last entry w/ the output
                    output = self.model.forward(x,Re)
                    output = torch.unsqueeze(output,1)
                    multi_step_output = torch.cat((multi_step_output,output),dim=1)

                # print('output shape: ',multi_step_output.shape)
                # update logs
                loss = self.loss_criteria(multi_step_output,y,mask)
                self.loss_vec.extend([loss.data.cpu().detach().numpy()])
                # print('other output: ',loss.shape,loss,type(loss))
                #backprop
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() # zero out the gradient

                updates += 1
                if updates > self.num_updates:
                    break

                if idx%10 == 0:
                    print(idx)
    
        return self.loss_vec