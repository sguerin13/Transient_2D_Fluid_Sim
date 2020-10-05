'''
Class to handle all of the training tasks for a model with multi-step loss

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

class Trainer_MultiStep():

    def __init__(self,model,scalers,file_path,model_file_name=None,board_file = None):
        self.model = model # NN model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_file_name = model_file_name # if we are saving the best model
        self.file_path = file_path # file path of the data_set
        self.scalers = scalers
        self.dataset = XYDataset(path=self.file_path,scalers=self.scalers)
        self.board_file = board_file

    def setup_training(self,loss_criteria,optimizer,
                       epochs,lr,min_eps,
                       val_split=None,loss_steps = 2,
                       batch_size = 24):
        self.lr = lr
        self.loss_criteria = loss_criteria # MaskedLoss(nn.MSELoss())
        self.optimizer = optimizer(self.model.parameters(),lr=lr) #torch.optim.Adam(self.model.parameters(),lr = lr)
        self.epochs = epochs
        self.val_split = val_split
        self.batch_size = batch_size
        self.loss_steps = loss_steps
        self.min_eps = min_eps
        self.writer=None
        if self.board_file is not None:
            self.writer = SummaryWriter('runs/'+self.board_file)

    def split_data(self):
        # borrowed from https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
        data_len = len(self.dataset)
        indices = list(range(data_len))
        split = int(np.floor(self.val_split*data_len))
        
        # split out the training and validation sets
        self.train_set,self.val_set = random_split(self.dataset,(data_len-split,split))
        del data_len,indices,split

    def build_data_loaders(self):
        if self.val_split is not None:
            self.split_data() # creates train and validation dataset
            self.dloader = DataLoader(self.train_set,batch_size=self.batch_size,
                                      pin_memory=False,shuffle=True,drop_last=False)
            self.val_dloader = DataLoader(self.val_set,batch_size=self.batch_size,
                                          pin_memory=False,shuffle=True,drop_last=False)
        else:
            self.dloader = DataLoader(self.dataset,batch_size=self.batch_size,pin_memory=False,
                                      drop_last=False)

    def _grab_data(self,data):           
        # send data to gpu
        x    = data[0].to(self.device).float()
        Re   = data[1].to(self.device).float()
        mask = data[2].to(self.device).float()
        y    = data[3].to(self.device).float()

        mask = torch.unsqueeze(mask,dim=1)
        if self.loss_steps == 2:
            mask = torch.cat((mask,mask),dim=1)
        elif self.loss_steps == 3:
            mask = torch.cat((mask,mask,mask),dim==1)
        else:
            pass
        return x,Re,mask,y

    def _update_logs(self,loss,iterr,idx,mode='train'):
        if mode == 'train':
            self.loss_vec.extend([loss.data.cpu().detach().numpy()])
            if self.board_file is not None:
                if (idx % 10 == 0) and (idx != 0):
                    self.writer.add_scalar('training_loss',np.sum(self.loss_vec[-10:])/10,iterr*10)
                    self.writer.flush()
                    iterr +=1
            return iterr
        
        else: # val_mode
            self.val_loss_vec.extend([loss.data.cpu().detach().numpy()])
            if self.board_file is not None:
                if (idx % 10 == 0) and (idx != 0):
                    self.writer.add_scalar('val_loss',np.sum(self.val_loss_vec[-10:])/10,iterr*10)
                    self.writer.flush()
                    iterr += 1
            return iterr

    def _save_model(self):
        sub_folder = str.split(self.model_file_name,'.pth')[0]
        if not os.path.isdir('./saved_models/'+sub_folder):
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

    def _update_screen(self,eps,idx,start_time):
        end_time = time.time()
        run_time = end_time - start_time
        print(
        'Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}, time: {:.3f}'.
        format(eps, (idx+1) * self.batch_size, len(self.dloader.dataset),
            100. * (idx+1) / len(self.dloader), self.loss_vec[-1],run_time))


    def train_net(self,save_model=False):
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
                x,Re,mask,y = self._grab_data(data)

                # forward pass
                output = self.model.forward(x,Re)
                output = torch.unsqueeze(output,1)
                multi_step_output = output #place holder

                for _ in range(1,self.loss_steps):
                    # print(x.shape,x[:,-1,:-1,:,:].shape,output.shape)
                    x[:,-1,:-1,:,:] = output.view(output.shape[0],output.shape[2],
                                                  output.shape[3],output.shape[4])     # replace vx, vy of last entry w/ the output
                    output = self.model.forward(x,Re)
                    output = torch.unsqueeze(output,1)
                    multi_step_output = torch.cat((multi_step_output,output),dim=1)

                # calc loss / update logs
                loss = self.loss_criteria(multi_step_output,y,mask)
                iterr = self._update_logs(loss=loss,
                                          iterr=iterr, idx=idx,mode='train')
                #backprop
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # print to the screen and free up some memory
                self._update_screen(eps=eps,idx=idx,start_time=start_time)
                del x,y,mask,Re

            # validation set
            if self.val_split is not None:
                self.model.eval()
                torch.set_grad_enabled(False)
                
                for v_idx,(v_data) in enumerate(self.val_dloader):
                    x,Re,mask,y = self._grab_data(v_data)
                    
                    # forward pass
                    output = self.model.forward(x,Re)
                    output = torch.unsqueeze(output,1)
                    multi_step_output = output #place holder
                    for _ in range(1,self.loss_steps):
                        x[:,-1,:-1,:,:] = output.view(output.shape[0],output.shape[2],
                                                      output.shape[3],output.shape[4])    # replace vx, vy of last entry w/ the output
                        output = self.model.forward(x,Re)
                        output = torch.unsqueeze(output,1)
                        multi_step_output = torch.cat((multi_step_output,output),dim=1)

                    val_loss = self.loss_criteria(output,y,mask)
                    v_iterr = self._update_logs(loss=val_loss,
                                                iterr=v_iterr,idx=v_idx,mode='val')

                print('val_loss: ',val_loss)
            print('Best Loss:', self.best_loss)

            # check model performance
            val_loss_avg = np.mean(self.val_loss_vec[-1000:])
            if val_loss_avg <= self.best_loss:
                self.best_loss = val_loss_avg
                if save_model==True:
                    self.best_model_state = self.model.state_dict()
            elif eps < self.min_eps:
                pass
            else:
                # we are not improving
                print('model not improving, exiting training \n')
                break #sends us to save model
      
        if save_model == True and self.model_file_name is not None:
            self._save_model()

