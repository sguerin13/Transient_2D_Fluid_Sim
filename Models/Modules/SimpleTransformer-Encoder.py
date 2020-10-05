import torch
import torch.nn as nn

import numpy as np



class TransformerEncoderNet(nn.Module):
    def __init__(self,num_heads=4,ff_dim=512):
        super(TransformerEncoderNet,self).__init__()

        self.num_heads = num_heads
        self.ff_dim = ff_dim

        # d_model is the input dimension
        # n_heads is the number of attention heads
        # dim_feedforward is the size of the FF net within the encoder
        self.enc_layer = nn.TransformerEncoderLayer(d_model = 512,nhead = self.num_heads,dim_feedforward=self.ff_dim, dropout=0,activation = 'relu')
        
        # need to pass the call the layer definition, the number of encoder layers
        # didnt include the layer norm here
        self.AttnEncoder = nn.TransformerEncoder(self.enc_layer,num_layers=2)                                         

    def forward(self):
        '''
        inputs to the attn encoder are:
            - src: the source sequence is:
                - size (S,N,E)
                    - S is the sequence length
                    - N is the Batch Size
                    - E is the number of features aka input vector dimension

            - mask: a sequence mask for when certain values should be ignored
                - size (S,S)
            
        '''


        vec_in = torch.rand(10,32,512)
        mask = torch.rand(10,10)
        for i in range(10):
            for j in range(10):
                if i<=j:
                    mask[i,j] = 1
                else:
                    mask[i,j] = 0

        

        y = self.AttnEncoder(vec_in,mask) # also returns hidden/context weight values, but we don't care about that
        return y,mask


