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

ROOT_IWR = '/export/data/ffeiden/data/pointodyssey' 
ROOT_HELIX = None

if os.path.exists(ROOT_IWR):
    ROOT = ROOT_IWR
else:
    ROOT = ROOT_HELIX

class PointOdyssey(SyntheticDepthDataset):
    def __init__(self,
                 augmenter=None,
                 is_test=False,

                 root=ROOT,
                 sample_length=1,
                 is_val=False,

                 verbose=False,
                 ):
        super().__init__(augmenter=augmenter, is_test=is_test)
        
        self.root = root
        self.sample_length = sample_length
        self.is_val = is_val
        assert not (self.is_test & self.is_val)

        self.verbose=verbose
        self.max_depth = 1_000. # m
        self.min_depth = 0. # m
        self.flip_to_open3d_coord = np.diag([1., 1., 1., 1.])
        self.Cam_to_World = False # This means extrinsics are Cam_to_World

        # load paths
        # TODO: Handle sampling of different scene lengths use sample_length & skip_frames
        # The question is: do we want to use static sample_length or dynamic; same for skip_frames 
        # If dynaic we leave this and sample indexe during getitem dynamicly. 
        # Wenn nicht dann kann das auch schon jetzt berechnet werden.
        if self.is_test:
            self.scenes = [entry.name for entry in os.scandir(os.path.join(self.root, 'test')) if entry.is_dir()]
        elif self.is_val:
            self.scenes = [entry.name for entry in os.scandir(os.path.join(self.root, 'val')) if entry.is_dir()]
        else:
            self.scenes = [entry.name for entry in os.scandir(os.path.join(self.root, 'train')) if entry.is_dir()]
        
        self.scenes = natsorted(self.scenes)
        for scene in self.scenes:
            scene_dic = {}
            scene_dic['image'] = natsorted(glob(os.path.join(self.root, 'train', scene, 'rgbs', 'rgb_*.jpg')))
            scene_dic['depth'] = natsorted(glob(os.path.join(self.root, 'train', scene, 'depths', 'depth_*.png')))
            scene_dic['anno'] = os.path.join(self.root, 'train', scene, 'anno.npz')
            self.sample_list.append(scene_dic)
        
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
            image[idx] = torch.from_numpy(rearrange(image_tmp, 'h w c -> c h w'))
            idx_img = self.__extract_index__(image_path)
            idx_list.append(idx_img)
            if self.verbose:
                img_idx = self.__extract_index__(image_path)
                depth_idx = self.__extract_index__(sample_paths['depth'][idx])
                assert depth_idx == img_idx, f'Depth and Image Paths do not fit. Image: {image_path} Depth: {sample_paths['depth'][idx]}'
            
            # Load metric Depth
            depth_tmp = cv2.imread(sample_paths['depth'][idx], cv2.IMREAD_UNCHANGED).astype('float32')
            # Convert to metric Depth like stated in PointOdyssey Repo: https://github.com/y-zheng18/point_odyssey/issues/14
            depth[idx] = torch.from_numpy(depth_tmp / 65_535 * 1_000)
            valid_depth[idx] = torch.where((depth[idx] > self.min_depth) & (depth[idx] < self.max_depth), True, False) 
        
        # Load Camera Parameter
        annotations = np.load(sample_paths['anno'])
        Intrinsics = torch.from_numpy(annotations['intrinsics'][idx_list])
        Extrinsics = torch.from_numpy(annotations['extrinsics'][idx_list])
        
        # Return dictionary
        sample_data = {}
        sample_data['image'] = image
        sample_data['depth'] = depth 
        sample_data['valid_depth'] = valid_depth
        sample_data['intrinsics'] = Intrinsics
        sample_data['extrinsics'] = Extrinsics

        return sample_data
            


