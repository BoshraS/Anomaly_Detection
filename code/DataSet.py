import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import logging
import torchvision
import torchvision.transforms as transform
from typing import Callable, Union, Optional, Tuple, Dict, List
import tractogramReader as tr


class CustomStrmDataset(Dataset):
    
    def __init__(self, 
                 root,
                 list_IDs,
                 numberOfStreamlines=1000000, 
                 transformData=None, 
                 res=32, 
                 device='cuda',
                 seed=2022):
        """
        Initializes Dataset
        param: strm_dir: Path to dataset which all streamlines from all subjects are located
        param: logger: Logger to log Dataset operations to
        param: normalize: Whether streamlines should be normalized.
        """
        super(CustomStrmDataset, self).__init__()
        
        self.transform = transformData
        self.res = res
        self.trk = tr.T(root.encode("utf-8"))
        self.len = numberOfStreamlines
        self.list_IDs = list_IDs
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx): 

        ID = self.list_IDs[idx]
        streamline = np.asarray(self.trk.read(ID))                
        streamline = streamline.astype(np.float32)
        streamline = streamline.transpose(1, 0)

        if  self.transform:
            streamline = torch.from_numpy(streamline).type(torch.FloatTensor)
            
        return streamline

            