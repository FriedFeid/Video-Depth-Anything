import os
import copy
import torch
import numpy as np 
import torch.utils.data as data 
import torch.nn.functional as F
import cv2

from glob import glob
from natsort import natsorted
from einops import rearrange

from datasets.utils import SyntheticDepthDataset

ROOT_IWR = '/export/data/ffeiden/data/vkitti/'
ROOT_HELIX = '/gpfs/lsdf02/sd23g007/datasets/vkitti'

if os.path.exists(ROOT_IWR):
    ROOT = ROOT_IWR
else:
    ROOT = ROOT_HELIX

class VKITTI(SyntheticDepthDataset):
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
        self.max_depth = 655.35 # m
        self.min_depth = 0. # m
        self.flip_to_open3d_coord = np.diag([1., -1., -1., 1.])
        self.Cam_to_World = False # This means extrinsics are Cam_to_World

        # Load paths and save in scene list dict
        if self.is_test:
            NotImplementedError('VKITTI does not contain an official Test Splitt')
        if self.is_val:
            NotImplementedError('VKITTI does not contain an official Validation Splitt')
        else:
            self.scenes_tmp = [entry.name for entry in os.scandir(os.path.join(self.root)) if 'Scene' in entry.name]
        
        self.scenes = []
        for scene in self.scenes_tmp:
            for condition in os.scandir(os.path.join(self.root, scene)):
                if condition.is_dir():
                    self.scenes.append(os.path.join(self.root, scene, condition))
        self.scenes = natsorted(self.scenes)

        for scene in self.scenes:
            scene_dic = {}
            scene_dic['image'] = natsorted(glob(os.path.join(scene, 'frames', 'rgb', 'Camera_0','rgb_*.jpg')))
            scene_dic['depth'] = natsorted(glob(os.path.join(scene, 'frames', 'depth', 'Camera_0','depth_*.png')))
            scene_dic['intrinsics_path'] = os.path.join(scene, 'intrinsic.txt')
            scene_dic['extrinsics_path'] = os.path.join(scene, 'extrinsic.txt')
            scene_dic['cameraID'] = str(0)
            self.sample_list.append(scene_dic)

            scene_dic_cam1 = {}
            scene_dic_cam1['image'] = natsorted(glob(os.path.join(scene, 'frames', 'rgb', 'Camera_1','rgb_*.jpg')))
            scene_dic_cam1['depth'] = natsorted(glob(os.path.join(scene, 'frames', 'depth', 'Camera_1','depth_*.png')))
            scene_dic_cam1['intrinsics_path'] = os.path.join(scene, 'intrinsic.txt')
            scene_dic_cam1['extrinsics_path'] = os.path.join(scene, 'extrinsic.txt')
            scene_dic_cam1['cameraID'] = str(1)
            self.sample_list.append(scene_dic_cam1)
    
    def _load_camera_parameters(self, parameter_path, cameraID):
        '''
        Retruns Numpy Array of shape: [frames, 3x3] for the intrinsics path
        and [frames, 4x4] for the extrinsics path. 

        :param parameter_path: Path to the scene .txt file 
        :type parameter_path: str
        :param cameraID: ID of the Camera. [0, 1]
        :type cameraID: str

        :return: In case of intrinsic: np.array[frames, 3x3] (Intrinsics)
                 In case of extrinsics: Tuple (np.array[frames, 4x4], np.array[frames, 4x4], np.array[frames, 4x4])
                        (Extrinsics, Rotation, Translation)
        :rtype: np.array
        '''
        if 'intrinsic.txt' in parameter_path:
            intrinsics = []
            with open (parameter_path, 'r') as file: 
                for i, line in enumerate(file):
                    if i == 0:
                        keys = line.split()
                    else:
                        # Values for intrinsics: 0 frame, 1 cameraID, 2 K[0,0], 3 K[1,1], 4 K[0,2], 5 K[1,2]
                        values = line.split()
                        if values[1] == cameraID:
                            intr_tmp = np.eye(3)
                            intr_tmp[0,0] = float(values[2])
                            intr_tmp[1,1] = float(values[3])
                            intr_tmp[0,2] = float(values[4])
                            intr_tmp[1,2] = float(values[5])
                            intrinsics.append(intr_tmp)
            return np.stack(intrinsics, axis=0)
        elif 'extrinsic.txt' in parameter_path:
            extrinsics = []
            translation = []
            rotation = []
            with open (parameter_path, 'r') as file: 
                for i, line in enumerate(file):
                    if i == 0:
                        keys = line.split()
                    else:
                        # Values for extrinsics: 0 frame, 1 cameraID
                        # 2: r1,1 3: r1,2 4: r1,3 5: t1 6: r2,1 7: r2,2 8: r2,3 
                        # 9: t2 10: r3,1 11: r3,2 12: r3,3 13: t3 
                        values = line.split()
                        if values[1] == cameraID:
                            translation_matrix = np.diag(np.array([1., 1., 1., 1.]))
                            translation_matrix[0, 3] = float(values[5])
                            translation_matrix[1, 3] = float(values[9])
                            translation_matrix[2, 3] = float(values[13])

                            translation.append(translation_matrix)

                            rotation_matrix = np.diag(np.array([1., 1., 1., 1.]))
                            rotation_matrix[0,0] = float(values[2])
                            rotation_matrix[0,1] = float(values[3])
                            rotation_matrix[0,2] = float(values[4])
                            
                            rotation_matrix[1,0] = float(values[6])
                            rotation_matrix[1,1] = float(values[7])
                            rotation_matrix[1,2] = float(values[8])
                            
                            rotation_matrix[2,0] = float(values[10])
                            rotation_matrix[2,1] = float(values[11])
                            rotation_matrix[2,2] = float(values[12])

                            rotation.append(rotation_matrix)

                            extrinsics_matrix = translation_matrix @ rotation_matrix
                            extrinsics.append(extrinsics_matrix)
            return np.stack(extrinsics, axis=0), np.stack(rotation, axis=0), np.stack(translation, axis=0)
            
        else:
            NotImplementedError('No Handeling of other .txt Files in VKITTI implemented')
    
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
            idx_list.append(idx_img)
            if self.verbose:
                img_idx = self.__extract_index__(image_path)
                depth_idx = self.__extract_index__(sample_paths['depth'][idx])
                assert depth_idx == img_idx, f'Depth and Image Paths do not fit. Image: {image_path} Depth: {sample_paths['depth'][idx]}'
            
            # Load metric Depth
            depth_tmp = cv2.imread(sample_paths['depth'][idx], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) # in cm
            depth[idx] = torch.from_numpy(depth_tmp / 100.) # To convert to meters
            # In VKITTI we got values between: 0 and 655.35 meters: https://www.changjiangcai.com/studynotes/2020-05-16-Virtual-KITTI-2-Dataset/
            valid_depth[idx] = torch.where((depth[idx] > self.min_depth) & (depth[idx] < self.max_depth), True, False)

        # Load Camera Parameters
        Intrinsics_tmp = self._load_camera_parameters(sample_paths['intrinsics_path'],
                                                        sample_paths['cameraID'])
        Extrinsics_tmp, _, _ = self._load_camera_parameters(sample_paths['extrinsics_path'],
                                                            sample_paths['cameraID'])
        Intrinsics = torch.from_numpy(Intrinsics_tmp[idx_list])
        Extrinsics = torch.from_numpy(Extrinsics_tmp[idx_list])
        
        # Return dictionary
        sample_data = {}
        sample_data['image'] = image
        sample_data['depth'] = depth 
        sample_data['valid_depth'] = valid_depth
        sample_data['intrinsics'] = Intrinsics
        sample_data['extrinsics'] = Extrinsics

        return sample_data