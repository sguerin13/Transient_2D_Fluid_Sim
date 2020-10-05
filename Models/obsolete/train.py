'''
script to train the encoder portion of the network

'''

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler as STD
import Utils.data_utils as du
# from EncoderDecoderSDF import Encoder,Decoder,EncoderDecoder
from SimpleEncDec import Encoder,Decoder,EncoderDecoder
import matplotlib.pyplot as plt


'''
Data Loading Chunk 

'''
# load data
sim,sdf,param_vec = du.load_array_with_sdf_and_params('Circle_Transient_5_7_2020_0_57.npz')# normalize the data

# scale data
sim_scaled,vscaler,_ = du.normalize_sim_data(sim,std=True)
sdf_scaled,_ = du.normalize_data(sdf,SDF=True)
# param_scaled = du.normalize_data(param_vec) # will use when there are more than 1 datapoint


### AUTOENCODER ROUTINE ###
# create multiple copies of the SDF and params so that they stack with the sim data

sdf_scaled = np.repeat(sdf_scaled,sim_scaled.shape[0],axis=0)
sdf_scaled = np.expand_dims(sdf_scaled,axis = 1)
conv_input = np.concatenate((sim_scaled,sdf_scaled),axis = 1) # input to conv_encoder

# sdf = np.repeat(sdf,sim_scaled.shape[0],axis=0)
# sdf = np.expand_dims(sdf,axis = 1)
# conv_input = np.concatenate((sim,sdf),axis = 1) # input to conv_encoder

param_vec = np.expand_dims(param_vec,axis = 0)
param_vec = np.repeat(param_vec,sim_scaled.shape[0],axis=0)
target = conv_input[:,:-1,:,:]

plt.imshow(sdf_scaled[0,0,:,:])
plt.show()

# conv_input = conv_input[0]
# conv_input = np.expand_dims(conv_input,0)
# target = target[0]
# target = np.expand_dims(target,0)
# param_vec = param_vec[0]
# param_vec = np.expand_dims(param_vec,0)

train_data = du.np_to_torch_dataloader(conv_input,target,Params=param_vec,batch = 20)
# enc = Encoder().double()
# dec = Decoder().double()
model = EncoderDecoder().double()
model.cuda()

model.train_net(train_data,epochs=1000,lr = .002)

plt.figure(figsize = (20,10))
plt.plot(model.loss_vec)
plt.title('training loss')
plt.xlabel('steps')
plt.ylabel('loss')
plt.yscale('log')
plt.show()
file_str = 'conv_enc_dec_simple_bottleneck-5-10'
torch.save(model.state_dict(),file_str)

### test output ### 
import matplotlib.pyplot as plt
x_demo = torch.from_numpy(conv_input[0])
x_demo = x_demo.unsqueeze(0)
param_demo = torch.from_numpy(param_vec[0])
param_demo = param_demo.view((1,3))
model.cpu()
model.eval()
y = model.forward(x_demo,param_demo)

y_numpy = y.detach().numpy()

plt.figure(figsize=(10,10))
plt.suptitle('Density',y=.62)
plt.subplot(1,3,1)
plt.imshow(target[0,2,:,:])
plt.title('Target')
plt.subplot(1,3,2)
plt.imshow(y_numpy[0,2,:,:],vmin=np.min(target[0,2,:,:]),vmax=np.max(target[0,2,:,:]))
plt.title('Prediction')
plt.subplot(1,3,3)
plt.imshow(target[0,2,:,:] - y_numpy[0,2,:,:])
plt.title('Difference')
plt.show()
