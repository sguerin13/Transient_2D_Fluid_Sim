import torch
import torch.nn as nn



class SimpleLSTM(nn.Module):
    def __init__(self,input_dim = 512,hidden_dim = 512,num_layers = 2):
        super(SimpleLSTM,self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = nn.LSTM(input_size = self.input_dim,hidden_size = self.hidden_dim,num_layers = self.num_layers)

    def forward(self,x):

        # inputs into the lstm should be of the following form
        # (seq_length,batch_size,input_vector)
        # or (batch_size,seq_length,input_vector) if batch_first=True

        full_seq,(output,_) = self.model(x) # also returns hidden/context weight values, but we don't care about that
        return output,full_seq
