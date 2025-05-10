import os 
import copy 
import torch
import numpy as np 
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import re
import sys

import OpenEXR
import Imath

from glob import glob 
from natsort import natsorted
from einops import rearrange
from scipy.spatial.transform import Rotation as R

from datasets.utils import SyntheticDepthDataset

ROOT_IWR = '/export/scratch/vislearn/feiden/IRS/' 
ROOT_HELIX = None

if os.path.exists(ROOT_IWR):
    ROOT = ROOT_IWR
else:
    ROOT = ROOT_HELIX


# Helper Funktions 
def exr2hdr(exrpath):
    File = OpenEXR.InputFile(exrpath)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if (CNum > 1):
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1],Size[0],CNum),dtype=np.float32)
    if (CNum == 1):
        hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
    else:
        hdr[:,:,0] = np.reshape(Pixels[0],(Size[1],Size[0]))
        hdr[:,:,1] = np.reshape(Pixels[1],(Size[1],Size[0]))
        hdr[:,:,2] = np.reshape(Pixels[2],(Size[1],Size[0]))
    return hdr

def load_exr(filename):
	hdr = exr2hdr(filename)
	h, w, c = hdr.shape
	if c == 1:
		hdr = np.squeeze(hdr)
	return hdr

# def load_exr(filename):
#     hdr = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    
#     if hdr is None:
#         raise IOError(f"EXR-Datei konnte nicht geladen werden: {filename}")
    
#     if hdr.ndim == 2 or hdr.shape[2] == 1:
#         return np.squeeze(hdr)
#     return hdr

class IRS(SyntheticDepthDataset):
    def __init__(self, augmenter=None, 
                 is_test=False,
                 
                 root=ROOT,
                 sample_length=1,
                 is_val=False,
                 
                 verbose=False):
        super().__init__(augmenter, is_test)

        self.root = root 
        self.sample_length = sample_length
        self.is_val = is_val
        assert not (self.is_test & self.is_val)

        self.verbose = verbose
        self.max_depth = 100. # m There is no upper Limit in TartanAir for sky. This is just set by me by
        # Looking at 3d Point clouds
        self.min_depth = 0. # m
        self.flip_to_open3d_coord = np.array([[1,  0,  0, 0],
                                              [0,  1,  0, 0],
                                              [0,  0,  1, 0],
                                              [0,  0,  0, 1],
                                                ], dtype=np.float32)
        self.Cam_to_World = False # This means extrinsics are Cam_to_World

        # I assume IRS is always same Intrinsics but not sure 
        # Camera Center is only a guess by me. They do not provide this
        self.Intrinsics = torch.from_numpy(np.array([[480., 0., 480.],
                                                     [0., 480., 270.],
                                                     [0., 0., 1.],], dtype=np.float32))
        self.baseline = 0.1 # m (taken from another repository: https://github.com/HKBU-HPML/IRS)

        self.scenes = []
        for map in os.scandir(os.path.join(self.root)):
            if map.is_dir():
                if map.name != 'IRS_small':
                    for setup in os.scandir(os.path.join(self.root, map.name)):
                        if setup.is_dir():
                            self.scenes.append(os.path.join(self.root, map.name, setup.name))
        
        self.scenes = natsorted(self.scenes)
        for scene in self.scenes:
            # IRS only provides disparity for the LEFT image 
            # So we only load this
            scene_dic = {}
            scene_dic['image'] = natsorted(glob(os.path.join(scene, 'l_*.png')))
            scene_dic['disparity'] = natsorted(glob(os.path.join(scene, 'd_*.exr')))
            # scene_dic['pose_path'] = None # TODO: Not available, Missing in data
            self.sample_list.append(scene_dic)


    def _Pose_to_Extrinsics_(self, pose_path):
        '''
        transfer a camera traj to ned frame traj
        There is code here, which transforms to open3D:
        https://github.com/blackjack2015/IRS/pull/11/commits/6109bd05073419a900e9cb4ec8c21fa6c1125049
        '''
        return 0

    def _load_data_samples(self, sample_paths):
        '''
        Loads Data for individual dataset
        
        :param sample_paths: dictionary of all the paths to load
        :type sample_paths: Dict
        
        :return: Dictionary with the following keys:
                    
                    - image:       torch.Tensor()     min: 0, max:1,      dtype=float.32
                    
                    - depth:       torch.Tensor()     [Metrische Tiefe]   dtype=float.32
                    
                    - valid_depth: torch.Tensor()     Bools               dtype=bool
                    
                    - intrinsics:  torch.Tensor()     [3x3] Matrix        dtype=float.32
                    
                    - extrinsics:  torch.Tensor()     [4x4] Matrix        dtype=float.32
        :rtype: dict
        '''

        num_frames = len(sample_paths['image'])
        h, w, c = cv2.imread(sample_paths['image'][0]).shape
        # prepare torch tensor for loading. 
        image = torch.zeros((num_frames, c, h, w))
        depth = torch.zeros((num_frames, h, w))
        valid_depth = torch.zeros_like(depth).to(dtype=torch.bool)

        idx_list = []
        for idx, image_path in enumerate(sample_paths['image']):
            image_tmp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype('float32') / 255.
            idx_img = self.__extract_index__(image_path)
            image[idx] = torch.from_numpy(rearrange(image_tmp, 'h w c -> c h w'))
            
            if self.verbose:
                img_idx = self.__extract_index__(image_path)
                depth_idx = self.__extract_index__(sample_paths['disparity'][idx])
                assert depth_idx == img_idx, f'Depth and Image Paths do not fit. Image: {image_path} Depth: {sample_paths['depth'][idx]}'

            disparity = torch.from_numpy(load_exr(sample_paths['disparity'][idx])) # Follows official Repo:https://github.com/blackjack2015/IRS/blob/master/dataloader/EXRloader.py
            if self.verbose:
                zero_values = (disparity == 0.).sum()
                if zero_values != 0:
                    print(f'WARNING: There are {zero_values} zeros in the disparity')
            depth_tmp = torch.where(disparity != 0., (self.baseline * self.Intrinsics[0][0]) / disparity, 0.)
            depth[idx] = depth_tmp
            valid_depth[idx] = torch.where((depth[idx] > self.min_depth) & (depth[idx] < self.max_depth), True, False)
        
        sample_data = {}
        sample_data['image'] = image
        sample_data['depth'] = torch.clip(depth, 0., self.max_depth)
        sample_data['valid_depth'] = valid_depth
        sample_data['intrinsics'] = self.Intrinsics.unsqueeze(0).repeat(num_frames, 1, 1)
        sample_data['extrinsics'] = torch.from_numpy(self.flip_to_open3d_coord).unsqueeze(0).repeat(num_frames, 1, 1) #TODO: Not available

        return sample_data



            