import torch
import torch.nn as nn
import numpy as np
from   PositionalEncoding import PositionalEncoding
import math

class AttentionEncoder(nn.Module):
    def __init__(self,num_heads=4,d_model = 800,ff_dim=512,
                 n_encoder_layers = 2,seq_length = 5,
                 dropout = .1,device = 'gpu'):
        super(AttentionEncoder,self).__init__()

        # variable definitions
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dout = dropout
        self.d_model = d_model
        self.n_encoder_layers = n_encoder_layers
        self.seq_length = seq_length
        self.device = device

        # d_model is the input dimension, n_heads is the number of attention heads, dim_feedforward is the size of the FF net within the encoder
        self.enc_layer = nn.TransformerEncoderLayer(d_model = self.d_model,nhead = self.num_heads,dim_feedforward=self.ff_dim, dropout=self.dout,activation = 'relu')
        # need to pass the call the layer definition, the number of encoder layers
        
        # didnt include the layer norm here
        self.AttnEncoder = nn.TransformerEncoder(self.enc_layer,num_layers=2,norm=nn.LayerNorm(self.d_model))

        # define positional encoding                                      
        self.pos_enc = PositionalEncoding(d_model = self.d_model,max_len = self.seq_length,dropout = self.dout)

        # define the mask:
        self.mask = self._generate_square_subsequent_mask(self.seq_length)

    # pulled from pytorch tutorial
    def _generate_square_subsequent_mask(self, S):
        mask = (torch.triu(torch.ones(S, S)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        if self.device == 'gpu':
            return mask.cuda()
        else:
            return mask

    def forward(self,x):

        '''
        inputs to the attn encoder are:
            - src: the source sequence is:
                - size (S,N,E)
                    - S is the sequence length
                    - N is the Batch Size
                    - E is the number of features aka input vector dimension

            - mask: a sequence mask for when certain values should be ignored
                - size (S,S)

        - Need to introduce the positional encoding and provide the mask to the encoder
        - See if you want to use the sqrt of the ff dim there... as the normalization?
            
        '''

        # current batch is of type (batch_size, seq_size,vector) but need to swap to (seq,batch,vector)
        x = x.permute(1, 0, 2)

        # add the position encoding
        x = self.pos_enc(x) * math.sqrt(self.d_model)

        # pass through to get the memory
        y = self.AttnEncoder(x,self.mask) # also returns hidden/context weight values, but we don't care about that
        
        # swap back into batch, seq, vector
        y = y.permute(1,0,2)
        return y


