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
import fnmatch

from datasets.utils import SyntheticDepthDataset

ROOT_IWR = '/export/data/ffeiden/data/SceneFlow' 
ROOT_HELIX = None

if os.path.exists(ROOT_IWR):
    ROOT = ROOT_IWR
else:
    ROOT = ROOT_HELIX

# Datasets Paths 
# Driving: *_focallength
# Monkaa: *_x2
# FlyingThings: TEST, TRAIN

class SceneFlow(SyntheticDepthDataset):
    def __init__(self, augmenter=None,
                 is_test=False,
                 
                 root=ROOT,
                 sample_length=1,
                 is_val=False,
                 
                 verbose=False,
                 use_flyThings=True, 
                 use_Driving=True,
                 use_Monkaa=True,):
        super().__init__(augmenter=augmenter, is_test=is_test)

        self.root = root 
        self.sample_length = sample_length
        self.is_val = is_val
        assert not (self.is_test & self.is_val)

        self.verbose = verbose
        self.max_depth = 800. # m Did not find an upper limit, But in 3d this seems to be the number of the sky
        self.min_depth = 0. # m
        self.flip_to_open3d_coord = np.array([[1,  0,  0, 0],
                                              [0,  0,  -1, 0],
                                              [0,  1,  0, 0],
                                              [0,  0,  0, 1],
                                                ], dtype=np.float32)
        self.Cam_to_World = True # This means extrinsics are Cam_to_World

        # Load paths and save in scene list dict
        assert any([use_flyThings, use_Driving, use_Monkaa]), 'At least one Dataset must be selected. \
            One of these must be True: use_flyThings, use_Driving, use_Monkaa'
        
        self.use_flyThings = use_flyThings
        self.use_Driving = use_Driving
        self.use_Monkaa = use_Monkaa

        self.poss_datasets = natsorted([entry.name for entry in os.scandir(os.path.join(self.root, 'frames_cleanpass')) if entry.is_dir()])

        self.scenes = []
        if self.use_flyThings:
            if self.is_test:
                for entry in os.scandir(os.path.join(self.root, 'frames_cleanpass', 'TEST')):
                    if entry.is_dir():
                        for scene_number in os.scandir(os.path.join(self.root, 'frames_cleanpass', 'TEST', entry.name)):
                            if scene_number.is_dir():
                                self.scenes.append(os.path.join('TEST', entry.name, scene_number.name))
            elif self.is_val:
                NotImplementedError('SceneFlow FlyingThings does not contain an official Validation Splitt')
            else:
                for entry in os.scandir(os.path.join(self.root, 'frames_cleanpass', 'TRAIN')):
                    if entry.is_dir():
                        for scene_number in os.scandir(os.path.join(self.root, 'frames_cleanpass', 'TRAIN', entry.name)):
                            if scene_number.is_dir():
                                self.scenes.append(os.path.join('TRAIN', entry.name, scene_number.name))
        self.scenes = natsorted(self.scenes)

        if self.use_Driving:
            # We just make use of the slow Scene, because otherwise this is redundant. 
            # We can adjust for speed by only using a supsample of all frames. Fast is 1/3 of the slow frames (300 instead of 900)
            for name in self.poss_datasets:
                if fnmatch.fnmatch(name, '*_focallength'):
                    if self.is_test:
                        NotImplementedError('SceneFlow Driving does not contain an official Test Splitt')
                    elif self.is_val:
                        NotImplementedError('SceneFlow Driving does not contain an official Validation Splitt')
                    else:
                        for entry in os.scandir(os.path.join(self.root, 'frames_cleanpass', name)):
                            if entry.is_dir():
                                self.scenes.append(os.path.join(name, entry.name, 'slow'))

        if self.use_Monkaa:
            for name in self.poss_datasets:
                if fnmatch.fnmatch(name, '*_x2'):
                    if self.is_test:
                        NotImplementedError('SceneFlow Monkaa does not contain an official Test Splitt')
                    elif self.is_val:
                        NotImplementedError('SceneFlow Monkaa does not contain an official Validation Splitt')
                    else:
                        self.scenes.append(os.path.join(name))
        
        # save in scene list dir
        for scene in self.scenes:
            scene_dic = {}
            scene_dic['image'] = natsorted(glob(os.path.join(root, 'frames_cleanpass', scene, 'left', '*.png')))
            scene_dic['depth'] = natsorted(glob(os.path.join(root, 'disparity', scene, 'left', '*.pfm')))
            scene_dic['extrinsics_path'] = os.path.join(root, 'camera_data', scene, 'camera_data.txt')
            scene_dic['cameraID'] = 'left'
            self.sample_list.append(scene_dic)

            scene_dic_cam1 = {}
            scene_dic_cam1['image'] = natsorted(glob(os.path.join(root,'frames_cleanpass', scene, 'right', '*.png')))
            scene_dic_cam1['depth'] = natsorted(glob(os.path.join(root, 'disparity', scene, 'right', '*.pfm')))
            scene_dic_cam1['extrinsics_path'] = os.path.join(root, 'camera_data', scene, 'camera_data.txt')
            scene_dic_cam1['cameraID'] = 'right'
            self.sample_list.append(scene_dic_cam1)
    
    def parse_extrinsics_from_text(self, parameter_path, cameraID="left"):
        '''
        Reads in the Camera Extrinsics from the txt file. 

        :param parameter_path: Path to the camera_data.txt file of the scene
        :type parameter_path: str
        :param cameraID: If the paramerts for the left or right Camera should be read
        :type cameraID: str

        :returns: torch.Tensor with all Extrinscs for each frame (frames, 4x4)
        :rtype: torch.Tensor
        '''
        matrices = []
        current_frame = None
        use_prefix = 'L' if cameraID == 'left' else 'R'

        with open(parameter_path, 'r') as file:
            for i, line in enumerate(file):
                line = line.strip()
                if line.startswith("Frame"):
                    current_frame = int(line.split()[1])
                elif line.startswith(use_prefix):
                    parts = line.split()[1:]  # remove 'L' or 'R'
                    values = list(map(float, parts))
                    assert len(values) == 16, f"Expected 16 values, got {len(values)}"
                    mat = [values[i:i+4] for i in range(0, 16, 4)]
                    matrices.append(mat)

        return torch.Tensor(matrices)
    
    def readPFM(self, file):
        '''
        Reading of the disparity values as given by official documentation: 
        https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlow/assets/code/python_pfm.py
        Since we are using a newer python version (better than python 3.8) we need to adjust for different
        string and bytes handling. 
        '''
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_line = file.readline().decode("utf-8")
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', dim_line)
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale_line = file.readline().decode("utf-8").strip()
        scale = float(scale_line)
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale
    
    def __extract_index__(self, path):
        base = path.split('.')[0]
        idx_str = base.split('/')[-1]
        return int(idx_str)

    def _load_camera_parameters(self, parameter_path, cameraID):
        '''
        Retruns Numpy Array of shape: [frames, 3x3] for the intrinsics path
        and [frames, 4x4] for the extrinsics path. 

        :param parameter_path: Path to the scene .txt file 
        :type parameter_path: str
        :param cameraID: ID of the Camera. [0, 1]
        :type cameraID: str

        :return: torch.Tensor[frames, 3x3] (Intrinsics)
                 torch.Tensor[frames, 4x4] (Extrinsics)
        :rtype: torch.Tensor
        '''
        if '15mm_focallength' in parameter_path:
            Intrinsics = torch.Tensor([[450., 0., 479.5],
                                      [0., 450., 269.5],
                                      [0., 0., 1.]]).to(dtype=torch.float32)
        else:
            Intrinsics = torch.Tensor([[1050., 0., 479.5],
                                      [0., 1050., 269.5],
                                      [0., 0., 1.]]).to(torch.float32)
        
        Extrinsics = self.parse_extrinsics_from_text(parameter_path, cameraID)
        Intrinsics = Intrinsics.expand(len(Extrinsics), -1, -1)

        return Intrinsics, Extrinsics
    
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
        Intrinsics = torch.zeros((num_frames, 3, 3))
        Extrinsics = torch.zeros((num_frames, 4, 4))

        idx_list = []
        for idx, image_path in enumerate(sample_paths['image']):
        # Load image 
            image_tmp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype('float32') / 255. # Directly convert in range 
            idx_img = self.__extract_index__(image_path)
            image[idx] = torch.from_numpy(rearrange(image_tmp, 'h w c -> c h w'))
            if 'TRAIN' in image_path or 'TEST' in image_path:
                idx_list.append(idx_img-6)
            else:    
                idx_list.append(idx_img-1)
            if self.verbose:
                img_idx = self.__extract_index__(image_path)
                depth_idx = self.__extract_index__(sample_paths['depth'][idx])
                assert depth_idx == img_idx, f'Depth and Image Paths do not fit. Image: {image_path} Depth: {sample_paths['depth'][idx]}'
            
            # Load metric Depth
            disp_tmp, scale = self.readPFM(sample_paths['depth'][idx]) # This is disparity as given by the official documentation we 
                                                                 # tranform this to depth via: depth = focallength*baseline/disparity
                                                                 # with focallength out of Intrinsics, baseline = 1.0 
            if '15mm_focallength' in image_path:
                focallength = 450.
            else:
                focallength = 1050.
            if scale != 1.0:
                print('WARNING: Scale differs from 1.0')
            depth[idx] = torch.from_numpy(np.where(disp_tmp == 0., 0., focallength * 1.0 / (disp_tmp * scale)))
            valid_depth[idx] = torch.where((depth[idx] > self.min_depth) & (depth[idx] < self.max_depth), True, False)

        # Load Camera Parameters
        Intrinsics, Extrinsics = self._load_camera_parameters(sample_paths['extrinsics_path'],
                                                            sample_paths['cameraID'])
        
        # Return dictionary
        sample_data = {}
        sample_data['image'] = image
        sample_data['depth'] = depth 
        sample_data['valid_depth'] = valid_depth
        sample_data['intrinsics'] = Intrinsics[idx_list]
        sample_data['extrinsics'] = Extrinsics[idx_list]

        return sample_data

