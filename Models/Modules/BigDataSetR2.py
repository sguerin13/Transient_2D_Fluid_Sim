'''
class for processing a large dataset
'''
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../')
import Utils.data_prep as dprep
import os

class XYDataset(Dataset):
    def __init__(self,path,scalers):
        '''
        - provide the path to the files
        - provide list containing the standard scalers
        - indicate the xy pair type, (1-1, many-one,etc.)
        '''
        self.path = path
        self.data_files = os.listdir(path)
        self.scalers = scalers

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data_path =self.path + self.data_files[idx]
        sample = torch.load(data_path)
        return sample

    def __len__(self):
        return len(self.data_files)



class SeqXYDataset(Dataset):
    def __init__(self,path,scalers):
        '''
        - provide the path to the files
        - provide list containing the standard scalers
        - indicate the xy pair type, (1-1, many-one,etc.)
        '''
        self.path = path
        self.data_files = os.listdir(path)
        self.scalers = scalers

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data_path =self.path + self.data_files[idx]
        sample = torch.load(data_path)
        return sample

    def __len__(self):
        return len(self.data_files)