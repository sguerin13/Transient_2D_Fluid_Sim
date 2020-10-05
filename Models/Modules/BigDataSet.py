'''
class for processing a large dataset
'''
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../')
import Utils.data_prep as dprep
import os

class BigAssDataset(Dataset):
    def __init__(self,path,scalers,xy_type = 'x to y'):
        '''
        - provide the path to the files
        - provide list containing the standard scalers
        - indicate the xy pair type, (1-1, many-one,etc.)

        '''
        self.path = path
        self.data_files = os.listdir(path)
        self.scalers = scalers
        self.xy_type = xy_type

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data_path =self.path + self.data_files[idx]
        sample = dprep.build_scaled_sample(data_path,self.scalers[0],self.scalers[1],
                                           self.scalers[2],self.scalers[3],self.xy_type)
        return sample

    def __len__(self):
        return len(self.data_files)