'''
Handles all of the data preprocessing for the train.py script

makes use of data utils to load files and prepare it for loading into the model

- load data
- standardize the data
- load it into batches for the model
- convert it into a torch data loader file

'''
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler as STD
from . import data_utils as du
import os


def build_scalers(path):

    '''
    - provide the path to the simulation directory

    - saves scalars built on the dataset in the file directory

    '''
    list_of_files = os.listdir(path = path+'.')
    # sim_scaled,vscaler,rho_scaler = du.normalize_sim_data(sim,std=True)
    # sdf_scaled,sdf_scaler = du.normalize_data(sdf,SDF=True)

    velo_scaler  = STD()
    rho_scaler   = STD()
    sdf_scaler   = STD()
    # param_scaler = STD()
    Re_scaler    = STD()

    for i in range(len(list_of_files)):
        # print(i)
        sim,sdf,mask,Re     = du.load_array_with_sdf_mask_Re(path + list_of_files[i])
        velo_scaler,rho_scaler = du.partially_fit_sim_scaler(velo_scaler,rho_scaler,sim)
        sdf_scaler             = du.partially_fit_data_scaler(sdf_scaler,sdf,SDF=True)
        # param_scaler           = du.partially_fit_data_scaler(param_scaler,param_vec,SDF=False)
        Re_scaler              = du.partially_fit_data_scaler(Re_scaler,Re,SDF=False)
        
    # save the scalers
    du.save_std_scaler(velo_scaler,'velo_scaler_100.pkl')
    du.save_std_scaler(rho_scaler,'rho_scaler_100.pkl')
    du.save_std_scaler(sdf_scaler,'sdf_scaler_100.pkl')
    # du.save_std_scaler(param_scaler,'./Data/scalers/param_scaler_circles_5-10-20.pkl')
    du.save_std_scaler(Re_scaler,'Re_scaler_100.pkl')
    
    return




def scale_data(data,scaler_1,scaler_2=None,data_type = 'sim'):

    '''
    scales a given array using provided scalers

    '''
    
    if data_type == 'sim':
        # scaler_1 = velo_scaler
        # scaler_2 = rho_scaler
        scaled_data = du.normalize_sim_w_scaler(data,scaler_1,scaler_2)
    
    elif data_type == 'SDF':
        scaled_data = du.normalize_data_w_scaler(data,scaler_1,SDF=True)

    # elif data_type == 'params': 
    #     # parameter vector
    #     scaled_data = du.normalize_data_w_scaler(data,scaler_1,SDF=False)

    elif data_type == 'Re': 
        # parameter vector
        scaled_data = du.normalize_data_w_scaler(data,scaler_1,SDF=False)
    
    else:
        print('please provide a data type')
        return

    return scaled_data



def build_scaled_dataset(path,velo_scaler,rho_scaler,sdf_scaler,Re_scaler,mode = 'x to y',num_scenes='max'):

    # decide the number of scenes we are pull from
    if num_scenes is not 'max':   
        list_of_files = os.listdir(path = path+'.')
        assert len(list_of_files) >= num_scenes, 'not enough scenes in the directory'
    else:
        list_of_files = os.listdir(path = path+'.')
        num_scenes = len(list_of_files)
    
    # load and scale data
    for i in range(num_scenes):
        # load the data from the file
        sim,sdf,mask,Re = du.load_array_with_sdf_mask_Re(path + list_of_files[i])
        # scale the data
        sim_scaled = scale_data(data=sim,scaler_1 = velo_scaler,scaler_2 = rho_scaler,data_type='sim')
        sdf_scaled = scale_data(data=sdf,scaler_1 = sdf_scaler,data_type = 'SDF')
        Re = Re.reshape((1,-1)) # turn it into a 1x1 array
        Re_scaled = scale_data(data=Re,scaler_1 = Re_scaler,data_type = 'Re')
        # print('param_scaled shape',param_scaled.shape)

        # create multiple copies of the SDF and params so that they stack with the sim data
        sdf_scaled = np.repeat(sdf_scaled,sim_scaled.shape[0],axis=0)  # now shape is (num_time_steps,y,x)
        sdf_scaled = np.expand_dims(sdf_scaled,axis = 1)               # now shape is (num_time_steps,1,y,x) - now we can stack
        
        # stack the sdf and the sim
        stacked_sim = np.concatenate((sim_scaled,sdf_scaled),axis = 1) # input to conv_encoder
        

        # prepare x,y pairs based on the sequence type 1 to 1, many to 1, many to many, etc
        if mode == 'x to y':

            # split into x and y
            x_data,y_data = du.sim_to_x_y(stacked_sim)
            # remove the sdf from the y_data since we only want v_x,v_y, and rho as the targets
            y_data = y_data[:,:-1,:,:]

            # create multiple copies of the Re #
            Re_scaled = np.repeat(Re_scaled,x_data.shape[0],axis=0) # shape: (num_time_steps,1,num_params)

            # create multiple copies of the mask
            mask = np.repeat(mask,x_data.shape[0],axis = 0)
            # give it a 4th dimension
            mask = np.expand_dims(mask,axis=1)


            if i == 0:
                list_out = [x_data,Re_scaled,mask,y_data] # create the list
            else:
                list_out[0] = np.concatenate((list_out[0],x_data),axis=0) # start stacking scenes
                list_out[1] = np.concatenate((list_out[1],Re_scaled),axis=0)
                list_out[2] = np.concatenate((list_out[2],mask),axis=0)
                list_out[3] = np.concatenate((list_out[3],y_data),axis = 0)

        else:
            print('add more here ')
            return

    return list_out


def build_scaled_sample(file_path,velo_scaler,rho_scaler,sdf_scaler,Re_scaler,mode = 'x to y',seq_length=None):
    '''
    - given a compressed file it loads the file, scales it, and places it into training x-y pairs
    '''
    # load and scale data
    sim,sdf,mask,Re = du.load_array_with_sdf_mask_Re(file_path)
    # scale the data
    sim_scaled = scale_data(data=sim,scaler_1 = velo_scaler,scaler_2 = rho_scaler,data_type='sim')
    sdf_scaled = scale_data(data=sdf,scaler_1 = sdf_scaler,data_type = 'SDF')
    Re = Re.reshape((1,-1)) # turn it into a 1x1 array
    Re_scaled = scale_data(data=Re,scaler_1 = Re_scaler,data_type = 'Re')
    # print('param_scaled shape',param_scaled.shape)

    # create multiple copies of the SDF and params so that they stack with the sim data
    sdf_scaled = np.repeat(sdf_scaled,sim_scaled.shape[0],axis=0)  # now shape is (num_time_steps,y,x)
    sdf_scaled = np.expand_dims(sdf_scaled,axis = 1)               # now shape is (num_time_steps,1,y,x) - now we can stack
    
    # stack the sdf and the sim
    stacked_sim = np.concatenate((sim_scaled,sdf_scaled),axis = 1) # input to conv_encoder
    
    # prepare x,y pairs based on the sequence type 1 to 1, many to 1, many to many, etc
    if mode == 'x to y':

        # split into x and y
        x_data,y_data = du.sim_to_x_y(stacked_sim)
        # remove the sdf from the y_data since we only want v_x,v_y, and rho as the targets
        y_data = y_data[:,:-1,:,:]

        # create multiple copies of the Re #
        Re_scaled = np.repeat(Re_scaled,x_data.shape[0],axis=0) # shape: (num_time_steps,1,num_params)

        # create multiple copies of the mask
        mask = np.repeat(mask,x_data.shape[0],axis = 0)
        # give it a 4th dimension
        mask = np.expand_dims(mask,axis=1)

        list_out = [x_data,Re_scaled,mask,y_data] # create the list

    elif mode == 'seq x to y':
        # split into x and y
        x_data,y_data = du.sim_to_seqx_y(stacked_sim,length = seq_length)

        # create multiple copies of the Re # for each element in the sequence
        Re_scaled = np.repeat(Re_scaled,x_data.shape[1],axis=0) # shape: (seq)_length,1,num_params)
        # create copies of Re for each sample
        Re_scaled = np.expand_dims(Re_scaled,axis=0)
        Re_scaled = np.repeat(Re_scaled,x_data.shape[0],axis=0)

        # create multiple copies of the mask for each element of the sequence
        # shape (seq,120,60)
        mask = np.repeat(mask,x_data.shape[1],axis = 0)
        # give it a 4th dimension
        # shape (seq,1,120,60)
        mask = np.expand_dims(mask,axis=1)
        # give it the same number of elements repeat for each sample in x
        mask = np.repeat(mask,x_data.shape[0],axis=0)

        list_out = [x_data,Re_scaled,mask,y_data] # create the list

    
    elif mode == 'seq x to seq y':
        # split into x and y
        x_data,y_data = du.sim_to_seqx_seqy(stacked_sim,x_len=seq_length,y_len=Y_len,allow_empty = False)

        # create multiple copies of the Re # for each element in the sequence
        Re_scaled = np.repeat(Re_scaled,x_data.shape[1],axis=0) # shape: (seq)_length,1,num_params)
        # create copies of Re for each sample
        Re_scaled = np.expand_dims(Re_scaled,axis=0)
        Re_scaled = np.repeat(Re_scaled,x_data.shape[0],axis=0)

        # create multiple copies of the mask for each element of the sequence
        # shape (seq,120,60)
        mask = np.repeat(mask,x_data.shape[1],axis = 0)
        # give it a 4th dimension
        # shape (seq,1,120,60)
        mask = np.expand_dims(mask,axis=1)
        # give it the same number of elements repeat for each sample in x
        mask = np.repeat(mask,x_data.shape[0],axis=0)

        list_out = [x_data,Re_scaled,mask,y_data] # create the list

    else:
        print('add more here ')
        return

    return list_out




def build_scaled_sample_no_rho(file_path,velo_scaler,rho_scaler,sdf_scaler,Re_scaler,mode = 'x to y',seq_length=None,Y_len=None):
    '''
    - given a compressed file it loads the file, scales it, and places it into training x-y pairs
    '''
    # load and scale data
    sim,sdf,mask,Re = du.load_array_with_sdf_mask_Re(file_path)
    # scale the data
    sim_scaled = scale_data(data=sim,scaler_1 = velo_scaler,scaler_2 = rho_scaler,data_type='sim')
    sdf_scaled = scale_data(data=sdf,scaler_1 = sdf_scaler,data_type = 'SDF')
    Re = Re.reshape((1,-1)) # turn it into a 1x1 array
    Re_scaled = scale_data(data=Re,scaler_1 = Re_scaler,data_type = 'Re')
    # print('param_scaled shape',param_scaled.shape)

    # create multiple copies of the SDF and params so that they stack with the sim data
    sdf_scaled = np.repeat(sdf_scaled,sim_scaled.shape[0],axis=0)  # now shape is (num_time_steps,y,x)
    sdf_scaled = np.expand_dims(sdf_scaled,axis = 1)               # now shape is (num_time_steps,1,y,x) - now we can stack
    
    # stack the sdf and the sim

    '''
    here is where I get rid of Rho

    '''
    stacked_sim = np.concatenate((sim_scaled[:,:-1,:,:],sdf_scaled),axis = 1) # input to conv_encoder
    
    # prepare x,y pairs based on the sequence type 1 to 1, many to 1, many to many, etc
    if mode == 'x to y':

        # split into x and y
        x_data,y_data = du.sim_to_x_y(stacked_sim)
        # remove the sdf from the y_data since we only want v_x,v_y, and rho as the targets
        y_data = y_data[:,:-1,:,:]

        # create multiple copies of the Re #
        Re_scaled = np.repeat(Re_scaled,x_data.shape[0],axis=0) # shape: (num_time_steps,1,num_params)

        # create multiple copies of the mask
        mask = np.repeat(mask,x_data.shape[0],axis = 0)
        # give it a 4th dimension
        mask = np.expand_dims(mask,axis=1)

        list_out = [x_data,Re_scaled,mask,y_data] # create the list

    elif mode == 'seq x to y':
        # split into x and y
        x_data,y_data = du.sim_to_seqx_y(stacked_sim,length = seq_length)
        y_data = y_data[:,:-1,:,:]
        # create multiple copies of the Re # for each element in the sequence
        Re_scaled = np.repeat(Re_scaled,x_data.shape[1],axis=0) # shape: (seq)_length,1,num_params)
        # create copies of Re for each sample
        Re_scaled = np.expand_dims(Re_scaled,axis=0)
        Re_scaled = np.repeat(Re_scaled,x_data.shape[0],axis=0)

        # create multiple copies of the mask for each element of the sequence
        # shape (seq,120,60)
        # give it a 4th dimension
        # shape (seq,1,120,60)
        mask = np.expand_dims(mask,axis=1)
        # give it the same number of elements repeat for each sample in x
        mask = np.repeat(mask,x_data.shape[0],axis=0)

        list_out = [x_data,Re_scaled,mask,y_data] # create the list

    
    elif mode == 'seq x to seq y':
        # split into x and y
        x_data,y_data = du.sim_to_seqx_seqy(stacked_sim,x_len=seq_length,y_len=Y_len,allow_empty = False)
        y_data = y_data[:,:,:-1,:,:] # get rid of rho
        # create multiple copies of the Re # for each element in the sequence
        Re_scaled = np.repeat(Re_scaled,x_data.shape[1],axis=0) # shape: (seq)_length,1,num_params)
        # create copies of Re for each sample
        Re_scaled = np.expand_dims(Re_scaled,axis=0)
        Re_scaled = np.repeat(Re_scaled,x_data.shape[0],axis=0)

        # create multiple copies of the mask for each element of the sequence
        # shape (seq,120,60)
        mask = np.repeat(mask,x_data.shape[1],axis = 0)
        # give it a 4th dimension
        # shape (seq,1,120,60)
        mask = np.expand_dims(mask,axis=1)
        # give it the same number of elements repeat for each sample in x
        mask = np.repeat(mask,x_data.shape[0],axis=0)

        list_out = [x_data,Re_scaled,mask,y_data] # create the list

    else:
        print('add more here ')
        return

    return list_out



def concat_collate(batch):


    x = torch.cat([torch.as_tensor(elem[0]) for elem in batch],0)
    Re  = torch.cat([torch.as_tensor(elem[1]) for elem in batch],0)
    mask     = torch.cat([torch.as_tensor(elem[2]) for elem in batch],0)
    target   = torch.cat([torch.as_tensor(elem[3]) for elem in batch],0)

    return(x,Re,mask,target)
