import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from Modules.BaseTransformer import AttentionEncoder  # this is okay
from Modules.PositionalEncoding import PositionalEncoding # this is okay
from AttnEncDecRe import Encoder,Decoder 
import math

class ConvAttn(nn.Module):

    def __init__(self,seq_length=5,n_heads=5,embedding_dim = 800,
                 ff_dim = 512,n_encoder_layers = 2,
                 model_dropout = .1,device = 'gpu'):
        super(ConvAttn,self).__init__()

        # constant definitions
        self.seq_len = seq_length
        self.embed_dim = embedding_dim
        self.dout = model_dropout
        self.n_heads = n_heads
        self.ff_dim  = ff_dim
        self.n_encoder_layers = n_encoder_layers
        self.device = device

        # define the convolutional encoder and decoder
        self.enc = Encoder()
        self.dec = Decoder(ch_in = self.seq_len*self.embed_dim)

        # define the attention encoder
        self.attn = AttentionEncoder(seq_length=self.seq_len,d_model=self.embed_dim,
                                     n_encoder_layers = self.n_encoder_layers,num_heads=self.n_heads,
                                     ff_dim = self.ff_dim,dropout = self.dout,device = self.device)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self,maps,Re):
        # a mini-batch of data of the form:
        # batch,seq_len,input_dims will be reshaped into a single "large batch" 
        # and passed through the encoder, once passed through the encoder, it will be reshaped
        # back into the form (batch,seq_len,dims)

        # example input 15 x num_scenes x 3 x 60 x 120
        map_shape = maps.shape
        maps = maps.view(map_shape[0]*map_shape[1],
                         map_shape[2],map_shape[3],map_shape[4])    # 15*num_scenes x 3 x 60 x 120
        Re = Re.view(Re.shape[0]*Re.shape[1],Re.shape[2])           # 15*num_scenes x 1
        
        x  = self.enc(maps,Re)                                      # encoder returns flat vector (15*num_scenes,embedding_dim)
        x = x.view(map_shape[0],map_shape[1],-1)                    # (15,num_scenes,embedding_dim)
        x = self.attn(x)                                            # (15,num_scenes,ff_dim)
        # print(x.shape)
        x = x.reshape(x.shape[0],-1,1,1)                                   # (15,num_scenes*ff_dim) - format it for the convolutional decoder
        # print(x.shape)
        y = self.dec(x)
        
        return y

    def pass_data(self):

        x = torch.rand(1,5,3,60,120)
        Re = torch.rand(1,5,1)
        y = self.forward(x,Re)
        return y



    




