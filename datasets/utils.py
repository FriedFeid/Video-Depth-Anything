import os 
import copy 
import torch
import numpy as np 
import torch.utils.data as data
import torch.nn.functional as F
from glob import glob

from typing import List, Optional
from PIL import Image

class SyntheticDepthDataset(data.Dataset):
    def __init__(self, augmenter=None, is_test=False):
        '''
        Base Class of all Datasets. Should handle general stuff. 
        '''
        self.augmenter = None  # TODO: put in init Arguments for the Augmenter or the augmenter itself 
        
        self.is_test = False  # TODO: put in init Different behaviour if it is only for testing 
        self.sample_list = []  # The individual Dataclasses will initialise a dictionary for each scene with keys
                               # 'image': Image path; 
                               # 'disparity': Path to the disparity 
                               # 'depth': path to depth 
                               # 'valid_mask': boolean mask of valid gt pixels 
                               # 'cam': path to Intrinsics & Extrinsics 
        self.extra_info = []  # Extra augmentation if the dataset provides them 

    def _load_data_samples(self, sample_paths):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_paths = self.sample_list[idx]

        # Wenn wir hier die anzahl der frames beschränken dann nur für die keys: 
        # 'image', 'depth', 'cam', 'depth_scale', 'valid_depth', 'image_size'
        # 
        # nicht für 'anno', 'intrinsics_path', 'extriniscs_path', 'cameraID'
        # Da diese ganze arrays mit annotation für alle frames der scene enthalten

        if self.is_test:
            raise NotImplementedError # TODO:
            return 0
        
        sample_data = self._load_data_samples(sample_paths)
        
        sample_size = len(sample_data['image'])
        
        if self.augmenter is not None:
            raise NotImplementedError #TODO:
        
        sample = {}
        # TODO: reshaping and handeling scale & shift invariance 
        # Calculation on processed data 
        sample['image'] = sample_data['image']
        sample['depth'] = sample_data['depth']
        sample['valid_depth'] = sample_data['valid_depth']
        sample['intrinsics'] = sample_data['intrinsics']
        sample['extrinsics'] = sample_data['extrinsics']
        if 'depth_mask' in sample_data:
            sample['depth_mask'] = sample_data['depth_mask']

        return sample

    def __extract_index__(self, path):
        base = path.split('.')[0]
        idx_str = base.split('_')[-1]
        return int(idx_str)
