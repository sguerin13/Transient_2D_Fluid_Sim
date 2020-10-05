'''
Class to handle all of the training tasks for parameter search

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


class TrainerXYDOE():

    def __init__(self,model,scalers,file_path,model_file_name=None,board_file = None):
        self.model = model # NN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_file_name = model_file_name # if we are saving the best model
        self.file_path = file_path # file path of the data_set
        self.scalers = scalers
        self.dataset = XYDataset(path=self.file_path,scalers=self.scalers)
        self.board_file = board_file

    def setup_training(self,loss_criteria,optimizer,epochs,lr,val_split=None,batch_size = 1000,sub_epochs = None):
        self.lr = lr
        self.loss_criteria = loss_criteria # MaskedLoss(nn.MSELoss())
        self.optimizer = optimizer(self.model.parameters(),lr=lr) #torch.optim.Adam(self.model.parameters(),lr = lr)
        self.epochs = epochs
        self.val_split = val_split
        self.batch_size = batch_size
        self.sub_epochs = sub_epochs # if we are going to train for less than 1 epoch
        self.writer=None
        if self.board_file is not None:
            self.writer = SummaryWriter('runs/'+self.board_file)
            # self.writer.add_graph(self.model)
            # self.writer.close()
        return

    def split_data(self):
        '''
        split dataset into train and validation set

        '''
        # borrowed from https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
        data_len = len(self.dataset)
        indices = list(range(data_len))
        split = int(np.floor(self.val_split*data_len))
        
        # split out the training and validation sets
        self.train_set,self.val_set = random_split(self.dataset,(data_len-split,split))

        del data_len,indices,split


    def build_data_loaders(self):
        
        '''
        build the dataloader objects

        '''
        if self.val_split is not None:
            self.split_data() # creates train and validation dataset
            self.dloader = DataLoader(self.train_set,batch_size=self.batch_size,pin_memory=True,drop_last=False,shuffle=True)
            self.val_dloader = DataLoader(self.val_set,batch_size=int(self.batch_size/2),pin_memory=True,drop_last=False,shuffle=True)
        else:
            self.dloader = DataLoader(self.dataset,batch_size=self.batch_size,pin_memory=True,
                                      drop_last=False,shuffle=True)

    def train_net(self,save_model=False):

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=.2,patience =50)
        self.loss_vec = []
        if self.val_split is not None:
            self.val_loss_vec = []
        self.best_loss = 100
        iterr = 1
        v_iterr = 1

        for eps in range(self.epochs):
            # training set
            torch.set_grad_enabled(True)
            self.model.train()
            
            for idx,(data) in enumerate(self.dloader): 
                start_time = time.time()                   
                
                # send data to gpu

                x    = data[0].float().to(self.device)
                Re   = data[1].float().to(self.device)
                mask = data[2].float().to(self.device)
                y    = data[3].float().to(self.device)

                # forward pass
                output = self.model.forward(x,Re)
                loss   = self.loss_criteria(output,y,mask)
                # update logs
                self.loss_vec.extend([loss.data.cpu().detach().numpy()])
                if self.board_file is not None:
                    if (idx % 10 == 0) and (idx != 0):
                        self.writer.add_scalar('training_loss',np.sum(self.loss_vec[-10:])/10,iterr*10)
                        self.writer.flush()
                        iterr +=1

                #backprop
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() # zero out the gradient
                end_time = time.time()
                run_time = end_time - start_time
                print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}, time: {:.3f}'.
                format(eps, (idx+1) * self.batch_size, len(self.dloader.dataset),
                    100. * (idx+1) / len(self.dloader), self.loss_vec[-1],run_time))


                # see if we need to break out of the training loops in less than one epoch
                if self.epochs == 1 and self.sub_epochs is not None:
                    if ((idx+1)/len(self.dloader)) >= self.sub_epochs:
                        break


                # torch.cuda.empty_cache()
            # validation set
            if self.val_split is not None:

                self.model.eval()
                torch.set_grad_enabled(False)
                for v_idx,(v_data) in enumerate(self.val_dloader): 
                                              
                    x    = v_data[0].to(self.device)
                    Re   = v_data[1].to(self.device)
                    mask = v_data[2].to(self.device)
                    y    = v_data[3].to(self.device)

                    # forward pass
                    output   = self.model.forward(x,Re)
                    val_loss = self.loss_criteria(output,y,mask)
                    self.val_loss_vec.extend([val_loss.data.cpu().detach().numpy()])
                    
                    if self.board_file is not None:
                        if (v_idx % 10 == 0) and (v_idx != 0):
                            self.writer.add_scalar('val_loss',np.sum(self.val_loss_vec[-10:])/10,v_iterr*10)
                            self.writer.flush()
                            v_iterr +=1

                    # torch.cuda.empty_cache()
                print('val_loss: ',val_loss)
                    
            # scheduler.step(loss)
            # if val_split is not 0:
            #     print('val_loss: ',val_loss)
            # print('epoch',eps+1,'/',self.epochs,' loss: ',loss.data)
            print('Best Loss:', self.best_loss)

            if loss < self.best_loss:
                self.best_loss = loss
                if save_model==True:
                    self.best_model_state = self.model.state_dict()
                    

        if save_model == True and self.model_file_name is not None:
            
            # create directory
            sub_folder = str.split(self.model_file_name,'.pth')[0]
            os.mkdir('./saved_models/'+sub_folder)

            # pull state dict over to the cpu
            model_state_dict = {}
            for key, val in self.best_model_state.items():
                model_state_dict[key] = val.cpu()
            
            model_file_str = './saved_models/' + sub_folder + '/' + self.model_file_name
            torch.save(model_state_dict,model_file_str)
            
            # save the loss information
            tr_loss_str  = './saved_models/' + sub_folder + '/' + 'tr_loss.pth'
            torch.save(self.loss_vec,tr_loss_str)
            if self.val_split is not None:
                val_loss_str   = './saved_models/' + sub_folder + '/' + 'val_loss.pth'
                torch.save(self.val_loss_vec,val_loss_str)





# class TrainerSeqXY():

#     def __init__(self,model,scalers,file_path,model_file_name=None,board_file = None):
#         self.model = model # NN model
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_file_name = model_file_name # if we are saving the best model
#         self.file_path = file_path # file path of the data_set
#         self.scalers = scalers
#         self.dataset = XYDataset(path=self.file_path,scalers=self.scalers)
#         self.board_file = board_file

#     def setup_training(self,loss_criteria,optimizer,epochs,lr,val_split=None,batch_size = 1000):
#         self.lr = lr
#         self.loss_criteria = loss_criteria # MaskedLoss(nn.MSELoss())
#         self.optimizer = optimizer(self.model.parameters(),lr=lr) #torch.optim.Adam(self.model.parameters(),lr = lr)
#         self.epochs = epochs
#         self.val_split = val_split
#         self.batch_size = batch_size
#         self.writer=None
#         if self.board_file is not None:
#             self.writer = SummaryWriter('runs/'+self.board_file)
#             # self.writer.add_graph(self.model)
#             # self.writer.close()
#         return

#     def split_data(self):
#         '''
#         split dataset into train and validation set

#         '''
#         # borrowed from https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
#         data_len = len(self.dataset)
#         indices = list(range(data_len))
#         split = int(np.floor(self.val_split*data_len))
        
#         # split out the training and validation sets
#         self.train_set,self.val_set = random_split(self.dataset,(data_len-split,split))

#         del data_len,indices,split


#     def build_data_loaders(self):
        
#         '''
#         build the dataloader objects

#         '''
#         if self.val_split is not None:
#             self.split_data() # creates train and validation dataset
#             self.dloader = DataLoader(self.train_set,batch_size=self.batch_size,pin_memory=True,drop_last=False)
#             self.val_dloader = DataLoader(self.val_set,batch_size=int(self.batch_size/2),pin_memory=True,drop_last=False)
#         else:
#             self.dloader = DataLoader(self.dataset,batch_size=self.batch_size,pin_memory=True,
#                                       drop_last=False)

#     def train_net(self,save_model=False):

#         # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=.2,patience =50)
#         self.loss_vec = []
#         if self.val_split is not None:
#             self.val_loss_vec = []
#         self.best_loss = 100
#         iterr = 1
#         v_iterr = 1
#         for eps in range(self.epochs):
#             # training set
#             torch.set_grad_enabled(True)
#             self.model.train()
            
#             for idx,(data) in enumerate(self.dloader): 
#                 start_time = time.time()                   
                
#                 # send data to gpu
#                 x    = data[0].to(self.device)
#                 Re   = data[1].to(self.device)
#                 mask = data[2].to(self.device)
#                 y    = data[3].to(self.device)

#                 # forward pass
#                 output = self.model.forward(x,Re)
#                 loss   = self.loss_criteria(output,y,mask)
#                 # update logs
#                 self.loss_vec.extend([loss.data.cpu().detach().numpy()])
#                 if self.board_file is not None:
#                     if (idx % 10 == 0) and (idx != 0):
#                         self.writer.add_scalar('training_loss',np.sum(self.loss_vec[-10:])/10,iterr*10)
#                         self.writer.flush()
#                         iterr +=1

#                 #backprop
#                 loss.backward()
#                 self.optimizer.step()
#                 self.optimizer.zero_grad() # zero out the gradient
#                 end_time = time.time()
#                 run_time = end_time - start_time
#                 print(
#                 'Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}, time: {:.3f}'.
#                 format(eps, (idx+1) * self.batch_size, len(self.dloader.dataset),
#                     100. * (idx+1) / len(self.dloader), self.loss_vec[-1],run_time))
#                 # torch.cuda.empty_cache()
#             # validation set
#             if self.val_split is not None:

#                 self.model.eval()
#                 torch.set_grad_enabled(False)
#                 for v_idx,(v_data) in enumerate(self.val_dloader): 
                                              
#                     x    = v_data[0].to(self.device)
#                     Re   = v_data[1].to(self.device)
#                     mask = v_data[2].to(self.device)
#                     y    = v_data[3].to(self.device)

#                     # forward pass
#                     output   = self.model.forward(x,Re)
#                     val_loss = self.loss_criteria(output,y,mask)
#                     self.val_loss_vec.extend([val_loss.data.cpu().detach().numpy()])
                    
#                     if self.board_file is not None:
#                         if (v_idx % 10 == 0) and (v_idx != 0):
#                             self.writer.add_scalar('val_loss',np.sum(self.val_loss_vec[-10:])/10,v_iterr*10)
#                             self.writer.flush()
#                             v_iterr +=1

#                     # torch.cuda.empty_cache()
#                 print('val_loss: ',val_loss)
                    
#             # scheduler.step(loss)
#             # if val_split is not 0:
#             #     print('val_loss: ',val_loss)
#             # print('epoch',eps+1,'/',self.epochs,' loss: ',loss.data)
#             print('Best Loss:', self.best_loss)

#             if loss < self.best_loss:
#                 self.best_loss = loss
#                 if save_model==True:
#                     self.best_model_state = self.model.state_dict()
                    

#         if save_model == True and self.model_file_name is not None:
            
#             # create directory
#             sub_folder = str.split(self.model_file_name,'.pth')[0]
#             os.mkdir('./saved_models/'+sub_folder)

#             # pull state dict over to the cpu
#             model_state_dict = {}
#             for key, val in self.best_model_state.items():
#                 model_state_dict[key] = val.cpu()
            
#             model_file_str = './saved_models/' + sub_folder + '/' + self.model_file_name
#             torch.save(model_state_dict,model_file_str)
            
#             # save the loss information
#             tr_loss_str  = './saved_models/' + sub_folder + '/' + 'tr_loss.pth'
#             torch.save(self.loss_vec,tr_loss_str)
#             val_loss_str   = './saved_models/' + sub_folder + '/' + 'val_loss.pth'
#             torch.save(self.val_loss_vec,val_loss_str)