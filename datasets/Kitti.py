import os

import torch
from typing import List, Tuple
try: 
    from typing import Literal, Optional
except:
    from typing_extensions import Literal
    from typing import Optional

from glob import glob
from natsort import natsorted
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from einops import rearrange

from datasets.utils import SyntheticDepthDataset

DATA_ROOT_IWR = '/export/data/ffeiden/data/KITTI/'
DATA_ROOT_HELIX = '/gpfs/bwfor/work/ws/hd_ud441-lsd/data/kitti/'
if os.path.exists(DATA_ROOT_IWR):
    ROOT = DATA_ROOT_IWR
else:
    ROOT = DATA_ROOT_HELIX


class KITTI(SyntheticDepthDataset):

    def __init__(self, augmenter=None,
                 is_test=False,
                 
                 root=ROOT,
                 sample_length=1,
                 is_val=False,
                 
                 verbose=False,):
        super().__init__(augmenter=augmenter, is_test=is_test)

        self.root = root 
        self.sample_length = sample_length
        self.is_val = is_val
        assert not (self.is_test & self.is_val)

        self.verbose = verbose
        self.max_depth = 255.9 # m
        self.min_depth = 0. # m
        self.flip_to_open3d_coord = np.diag([1., -1., -1., 1.])
        self.Cam_to_World = False # This means extrinsics are Cam_to_World

        # Load paths and save in scene List dict 
        if self.is_test:
            NotImplementedError('KITTI does not contain an official Test Splitt')
        if self.is_val:
            mode = "val"
        else:
            mode = "train"

        self.scenes_tmp = [entry.name for entry in os.scandir(os.path.join(self.root, 'kitti_depth/data_depth_annotated/', mode)) if '_drive_' in entry.name]

        for scene in natsorted(self.scenes_tmp): 
            # Split into the data and drive
            date, drive = scene.split('_drive_')
            drive = 'drive_' + drive

            scene_dic = {}
            scene_dic['image'] = natsorted(glob(os.path.join(self.root, 'kitti_raw', date, date+'_'+drive, 'image_03', 'data', '*.png')))
            scene_dic['depth'] = natsorted(glob(os.path.join(self.root, 'kitti_depth/data_depth_annotated/', mode, scene, 'proj_depth', 'groundtruth', 'image_03', '*.png')))
            scene_dic['cam_path'] = os.path.join(self.root, 'kitti_raw', date)
            scene_dic['oxts_data'] = natsorted(glob(os.path.join(self.root, 'kitti_raw', date, date+'_'+drive, 'oxts', 'data', '*.txt')))
            scene_dic['cameraID'] = str(3)
            self.sample_list.append(scene_dic)

            scene_dic_cam2 = {}
            scene_dic_cam2['image'] = natsorted(glob(os.path.join(self.root, 'kitti_raw', date, date+'_'+drive, 'image_02', 'data', '*.png')))
            scene_dic_cam2['depth'] = natsorted(glob(os.path.join(self.root, 'kitti_depth/data_depth_annotated/', mode, scene, 'proj_depth', 'groundtruth', 'image_02', '*.png')))
            scene_dic_cam2['cam_path'] = os.path.join(self.root, 'kitti_raw', date)
            scene_dic_cam2['oxts_data'] = natsorted(glob(os.path.join(self.root, 'kitti_raw', date, date+'_'+drive, 'oxts', 'data', '*.txt')))
            scene_dic_cam2['cameraID'] = str(2)
            self.sample_list.append(scene_dic_cam2)
        
    def _read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary.
        
        source: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py#L68
        """
        data = {}

        with open(os.path.join(filepath, 'calib_cam_to_cam.txt'), 'r') as f:
            for line in f.readlines():
                try:
                    key, value = line.split(':', 1)
                except ValueError:
                    key, value = line.split(' ', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        
        velo_data = {}
        
        with open(os.path.join(filepath, 'calib_velo_to_cam.txt'), 'r') as f:
            for line in f.readlines():
                try:
                    key, value = line.split(':', 1)
                except ValueError:
                    key, value = line.split(' ', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    velo_data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        
        R = velo_data['R'].reshape(3, 3)
        t = velo_data['T'].reshape(3, 1)
        tmp_R_t = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
        
        
        # Now calculate Kamera Parameters: 
        Cam_param = {}
        Cam_param['T_cam0unrect_velo'] = tmp_R_t
        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(data['P_rect_00'], (3, 4))
        P_rect_10 = np.reshape(data['P_rect_01'], (3, 4))
        P_rect_20 = np.reshape(data['P_rect_02'], (3, 4))
        P_rect_30 = np.reshape(data['P_rect_03'], (3, 4))

        Cam_param['P_rect_00'] = P_rect_00
        Cam_param['P_rect_10'] = P_rect_10
        Cam_param['P_rect_20'] = P_rect_20
        Cam_param['P_rect_30'] = P_rect_30

        # Create 4x4 matrices from the rectifying rotation matrices
        R_rect_00 = np.eye(4)
        R_rect_00[0:3, 0:3] = np.reshape(data['R_rect_00'], (3, 3))
        R_rect_10 = np.eye(4)
        R_rect_10[0:3, 0:3] = np.reshape(data['R_rect_01'], (3, 3))
        R_rect_20 = np.eye(4)
        R_rect_20[0:3, 0:3] = np.reshape(data['R_rect_02'], (3, 3))
        R_rect_30 = np.eye(4)
        R_rect_30[0:3, 0:3] = np.reshape(data['R_rect_03'], (3, 3))

        Cam_param['R_rect_00'] = R_rect_00
        Cam_param['R_rect_10'] = R_rect_10
        Cam_param['R_rect_20'] = R_rect_20
        Cam_param['R_rect_30'] = R_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T0 = np.eye(4)
        T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        Cam_param['T_0_0'] = T0
        Cam_param['T_0_1'] = T1
        Cam_param['T_0_2'] = T2
        Cam_param['T_0_3'] = T3

        # Compute the velodyne to rectified camera coordinate transforms
        Cam_param['T_cam0_velo'] = T0.dot(R_rect_00.dot(Cam_param['T_cam0unrect_velo']))
        Cam_param['T_cam1_velo'] = T1.dot(R_rect_00.dot(Cam_param['T_cam0unrect_velo']))
        Cam_param['T_cam2_velo'] = T2.dot(R_rect_00.dot(Cam_param['T_cam0unrect_velo']))
        Cam_param['T_cam3_velo'] = T3.dot(R_rect_00.dot(Cam_param['T_cam0unrect_velo']))

        # Compute the camera intrinsics
        Cam_param['K_cam0'] = P_rect_00[0:3, 0:3]
        Cam_param['K_cam1'] = P_rect_10[0:3, 0:3]
        Cam_param['K_cam2'] = P_rect_20[0:3, 0:3]
        Cam_param['K_cam3'] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(Cam_param['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(Cam_param['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(Cam_param['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(Cam_param['T_cam3_velo']).dot(p_cam)

        Cam_param['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        Cam_param['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        return Cam_param
    
    def __extract_index__(self, path):
        base = path.split('.')[0]
        idx_str = base.split('/')[-1]
        return int(idx_str)

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
        num_frames = len(sample_paths['depth'])
        h, w, c = cv2.imread(sample_paths['depth'][0]).shape
        # prepare torch tensor for loading. 
        image = torch.zeros((num_frames, c, h, w))
        depth = torch.zeros((num_frames, h, w))
        valid_depth = torch.zeros_like(depth).to(dtype=torch.bool)
        Extrinsics = None # TODO: Not Implemented for KITTI yet 

        idx_list = []
        for idx, depth_path in enumerate(sample_paths['depth']):
        # The GT depth starts with image 05 and ends with 5 th last image. So we can not use the first an last 5 frames. 
        # Load image 
            idx_img = self.__extract_index__(depth_path)
            image_tmp = cv2.cvtColor(cv2.imread(sample_paths['image'][idx_img]), cv2.COLOR_BGR2RGB).astype('float32') / 255. # Directly convert in range 
            image[idx] = torch.from_numpy(rearrange(image_tmp, 'h w c -> c h w'))
            idx_list.append(idx_img)
            if self.verbose:
                img_idx = self.__extract_index__(sample_paths['image'][idx_img])
                depth_idx = self.__extract_index__(depth_path)
                assert depth_idx == img_idx, f'Depth and Image Paths do not fit. Image: {sample_paths['image'][idx_img]} Depth: {depth_path}'
            
            # Load metric Depth
            depth_tmp = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth[idx] = torch.from_numpy(depth_tmp / 256.) # To convert to meters
            # Conversion to meters (https://github.com/joseph-zhong/KITTI-devkit)
            valid_depth[idx] = torch.where((depth[idx] > self.min_depth) & (depth[idx] < self.max_depth), True, False)

        # Load Camera Parameters
        Cam_Scene_calibration = self._read_calib_file(sample_paths['cam_path'])
        
        Intrinsics = torch.from_numpy(Cam_Scene_calibration['K_cam'+str(sample_paths['cameraID'])])
        Extrinsics_tmp = None
        Extrinsics = None
        
        # Return dictionary
        sample_data = {}
        sample_data['image'] = image
        sample_data['depth'] = depth 
        sample_data['valid_depth'] = valid_depth
        sample_data['intrinsics'] = Intrinsics.unsqueeze(0).repeat(len(image), 1, 1)
        sample_data['extrinsics'] = Extrinsics

        return sample_data