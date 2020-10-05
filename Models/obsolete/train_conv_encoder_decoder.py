'''
script to train the encoder portion of the network

'''


import torch
import numpy as np
from sklearn.preprocessing import StandardScaler as STD
import Utils.data_utils as du
from EncoderDecoderSDF import Encoder,Decoder,EncoderDecoder


'''
Data Loading Chunk 

'''
sim,sdf,param_vec = du.load_array_with_sdf_and_params('stokes_triangle_4.npz')# normalize the data
sim_scaled,_,_ = du.normalize_sim_data(sim)
sdf_scaled,_ = du.normalize_data(sdf,SDF=True)



#param_scaled = du.normalize_data(param_vec) # will use when there are more than 1 datapoint
# create multiple copies of the SDF and params so that they stack with the sim data

sdf_scaled = np.repeat(sdf_scaled,sim_scaled.shape[0],axis=0)
sdf_scaled = np.expand_dims(sdf_scaled,axis = 1)
conv_input = np.concatenate((sim_scaled,sdf_scaled),axis = 1) # input to conv_encoder
param_vec = np.expand_dims(param_vec,axis = 0)
param_vec = np.repeat(param_vec,sim_scaled.shape[0],axis=0)
target = conv_input[:,:-1,:,:]

# conv_input = conv_input[0]
# conv_input = np.expand_dims(conv_input,0)
# target = target[0]
# target = np.expand_dims(target,0)
# param_vec = param_vec[0]
# param_vec = np.expand_dims(param_vec,0)

train_data = du.np_to_torch_dataloader(conv_input,target,Params=param_vec,batch = 5)

enc = Encoder().double()
dec = Decoder().double()
model = EncoderDecoder(enc,dec).double()

model.train(train_data,epochs=500,lr = 1e-3)

file_str = 'conv_enc_dec_demo_500'
torch.save(model.state_dict(),file_str)


### test output ### 
import matplotlib.pyplot as plt
x_demo = torch.from_numpy(conv_input[0])
x_demo = x_demo.unsqueeze(0)
param_demo = torch.from_numpy(param_vec[0])
param_demo = param_demo.view((1,3))

y = model(x_demo,param_demo)

y_numpy = y.detach().numpy()

plt.subplot(1,2,1)
plt.imshow(y_numpy[0,2,:,:])
plt.subplot(1,2,2)
plt.imshow(target[0,2,:,:])
plt.show()