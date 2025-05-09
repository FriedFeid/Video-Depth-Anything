import os 
import copy 
import torch
import numpy as np 
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import re
import sys

from glob import glob 
from natsort import natsorted
from einops import rearrange
from scipy.spatial.transform import Rotation as R

from datasets.utils import SyntheticDepthDataset

ROOT_IWR = '/export/data/vislearn/rother_subgroup/rother_datasets/TartanAir/' 
ROOT_HELIX = None

if os.path.exists(ROOT_IWR):
    ROOT = ROOT_IWR
else:
    ROOT = ROOT_HELIX

# Helper Functions out of original Repository
# https://github.com/castacks/tartanair_tools/blob/master/evaluation/transformation.py
def pos_quats2SE_matrices(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = []
    for quat in quat_datas:
        SO = R.from_quat(quat[3:7]).as_matrix()
        SE = np.eye(4)
        SE[0:3,0:3] = SO
        SE[0:3,3]   = quat[0:3]
        SEs.append(SE)
    return SEs


class TartanAir(SyntheticDepthDataset):
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
        self.max_depth = 800. # m There is no upper Limit in TartanAir for sky. This is just set by me by
        # Looking at 3d Point clouds
        self.min_depth = 0. # m
        self.flip_to_open3d_coord = np.array([[1,  0,  0, 0],
                                              [0,  1,  0, 0],
                                              [0,  0,  1, 0],
                                              [0,  0,  0, 1],
                                                ], dtype=np.float32)
        self.Cam_to_World = False # This means extrinsics are Cam_to_World

        # In TartanAir the Intrinsics are always the same
        self.Intrinsics = torch.from_numpy(np.array([[320., 0., 320.],
                                                     [0., 320., 240.],
                                                     [0., 0., 1.],], dtype=np.float32))

        self.scenes = []
        for map in os.scandir(os.path.join(self.root)):
            if map.is_dir():
                for setting in ['Hard', 'Easy']:
                    for cam_path in os.scandir(os.path.join(self.root, map.name, setting)):
                        if cam_path.is_dir():
                            self.scenes.append(os.path.join(self.root, map.name, setting, cam_path.name))
        
        self.scenes = natsorted(self.scenes)
        for scene in self.scenes:
            scene_dic = {}
            scene_dic['image'] = natsorted(glob(os.path.join(scene, 'image_left', '*.png')))
            scene_dic['depth'] = natsorted(glob(os.path.join(scene, 'depth_left', '*.npy')))
            scene_dic['pose_path'] = os.path.join(scene, 'pose_left.txt')
            scene_dic['cameraID'] = 'left'
            self.sample_list.append(scene_dic)

            scene_dic_right = {}
            scene_dic_right['image'] = natsorted(glob(os.path.join(scene, 'image_right', '*.png')))
            scene_dic_right['depth'] = natsorted(glob(os.path.join(scene, 'depth_right', '*.npy')))
            scene_dic_right['pose_path'] = os.path.join(scene, 'pose_right.txt')
            scene_dic_right['cameraID'] = 'right'
            self.sample_list.append(scene_dic_right)

    def __extract_index__(self, path):
        base = path.split('.')[0]
        idx_str = base.split('/')[-1].split('_')[-2]
        return int(idx_str)
    
    def __extract_depth_index__(self, path):
        base = path.split('.')[0]
        idx_str = base.split('/')[-1].split('_')[-3]
        return int(idx_str)
    
    def _Pose_to_Extrinsics_(self, pose_path):
        '''
        transfer a camera traj to ned frame traj
        Copied from https://github.com/castacks/tartanair_tools/blob/master/evaluation/trajectory_transform.py
        '''
        traj = np.loadtxt(pose_path)

        T = np.array([[0,0,1,0],
                    [1,0,0,0],
                    [0,1,0,0],
                    [0,0,0,1]], dtype=np.float32) 
        T_inv = np.linalg.inv(T)
        new_traj = []
        traj_ses = pos_quats2SE_matrices(np.array(traj))

        for tt in traj_ses:
            ttt=T.dot(tt).dot(T_inv)
            new_traj.append(ttt)
            
        return np.array(new_traj)

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
                depth_idx = self.__extract_depth_index__(sample_paths['depth'][idx])
                assert depth_idx == img_idx, f'Depth and Image Paths do not fit. Image: {image_path} Depth: {sample_paths['depth'][idx]}'

            depth[idx] = torch.from_numpy(np.load(sample_paths['depth'][idx])) # Follows documentation of tartan Air: https://github.com/castacks/tartanair_tools/blob/master/data_type.md 
            valid_depth[idx] = torch.where((depth[idx] > self.min_depth) & (depth[idx] < self.max_depth), True, False)
        
        sample_data = {}
        sample_data['image'] = image
        sample_data['depth'] = torch.clip(depth, 0., self.max_depth)
        sample_data['valid_depth'] = valid_depth
        sample_data['intrinsics'] = self.Intrinsics.unsqueeze(0).repeat(num_frames, 1, 1)
        sample_data['extrinsics'] = torch.from_numpy(self._Pose_to_Extrinsics_(sample_paths['pose_path']))

        return sample_data



            