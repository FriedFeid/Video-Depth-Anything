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

ROOT_IWR = '/export/data/ffeiden/data/Sintel' 
ROOT_HELIX = None

if os.path.exists(ROOT_IWR):
    ROOT = ROOT_IWR
else:
    ROOT = ROOT_HELIX

# Used for loading Sintel Depth and Camera Parameters
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

# Code copied from original Sintel Dataset: http://sintel.is.tue.mpg.de/downloads
def sintel_depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def sintel_cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N



class Sintel(SyntheticDepthDataset):
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
        self.max_depth = 10_000 # m  --> Set by me, No upper limit
        self.min_depth = 0. # m
        self.flip_to_open3d_coord = np.array([[1,  0,  0, 0],
                                              [0,  0,  1, 0],
                                              [0,  1,  0, 0],
                                              [0,  0,  0, 1],
                                                ], dtype=np.float32)
        self.Cam_to_World = False # This means extrinsics are Cam_to_World

        # load paths
        if self.is_test:
            self.scenes = [entry.name for entry in os.scandir(os.path.join(self.root, 'test', 'final')) if entry.is_dir()]
            self.split = 'test'
        elif self.is_val:
            NotImplementedError('Sintel does not contain an official Validation Splitt')
        else:
            self.scenes = [entry.name for entry in os.scandir(os.path.join(self.root, 'training', 'final')) if entry.is_dir()]
            self.split = 'training'
        
        self.scenes = natsorted(self.scenes)
        for scene in self.scenes:
            scene_dic = {}
            scene_dic['image'] = natsorted(glob(os.path.join(self.root, self.split, 'final', scene, 'frame_*.png')))
            scene_dic['depth'] = natsorted(glob(os.path.join(self.root, self.split, 'depth', scene,'frame_*.dpt')))
            scene_dic['cam'] = natsorted(glob(os.path.join(self.root, self.split, 'camdata_left', scene, 'frame_*.cam')))
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
        
        for idx, image_path in enumerate(sample_paths['image']):
            # Load image 
            image_tmp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype('float32') / 255. # Directly convert in range 
            image[idx] = torch.from_numpy(rearrange(image_tmp, 'h w c -> c h w'))
            if self.verbose:
                img_idx = self.__extract_index__(image_path)
                depth_idx = self.__extract_index__(sample_paths['depth'][idx])
                assert depth_idx == img_idx, f'Depth and Image Paths do not fit. Image: {image_path} Depth: {sample_paths['depth'][idx]}'
            
            # Load metric Depth
            depth_tmp = sintel_depth_read(sample_paths['depth'][idx]) # Depth is already in m: https://sintel-depth.csail.mit.edu/faq
            depth[idx] = torch.from_numpy(depth_tmp)
            valid_depth[idx] = torch.where((depth[idx] > self.min_depth) & (depth[idx] < self.max_depth), True, False) # As far as i Understand no upper limit.

            # Load Camera Parameter
            Intrinsics_tmp, Extrinsics_tmp = sintel_cam_read(sample_paths['cam'][idx])
            Intrinsics[idx], Extrinsics[idx, :3] = torch.from_numpy(Intrinsics_tmp), torch.from_numpy(Extrinsics_tmp)
            Extrinsics[idx, 3, 3] = 1.
        
        # Return dictionary
        sample_data = {}
        sample_data['image'] = image
        sample_data['depth'] = depth 
        sample_data['valid_depth'] = valid_depth
        sample_data['intrinsics'] = Intrinsics
        sample_data['extrinsics'] = Extrinsics

        return sample_data
    
