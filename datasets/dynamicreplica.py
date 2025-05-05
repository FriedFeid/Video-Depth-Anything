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
from PIL import Image
import gzip
import json

from datasets.utils import SyntheticDepthDataset

ROOT_IWR = '/export/scratch/ffeiden/data/dynamicReplica' 
ROOT_HELIX = None

if os.path.exists(ROOT_IWR):
    ROOT = ROOT_IWR
else:
    ROOT = ROOT_HELIX

# Copied from official repo: 
# https://github.com/facebookresearch/dynamic_stereo/blob/main/datasets/dynamic_stereo_datasets.py

def _load_16big_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth

# Copied and adapted form original Repo
# https://github.com/facebookresearch/dynamic_stereo/blob/main/datasets/dynamic_stereo_datasets.py
def _get_pytorch3d_camera(cam_dict, image_size, 
                          scale: float ):
    '''
    Loads Camera Parameters out of dictionary. 

    :param cam_dict: Dictionary loaded out of config
    :type cam_dict: Dict
    :param image_size: List of the image size used to calculate focal Length and principle point in px
    :type image_size: List
    :param scale: I guess its the depth scale also saved in json file 
    :type scale: float

    :return: focal length, principal point, Rotation Matrix (3x3), Translation Matrix (3)
    :type: torch.tensor
    '''
    
    assert cam_dict is not None
    # principal point and focal length
    principal_point = torch.tensor(
        cam_dict['principal_point'], dtype=torch.float
    )
    focal_length = torch.tensor(cam_dict['focal_length'], dtype=torch.float)

    half_image_size_wh_orig = (
        torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
    )

    # first, we convert from the dataset's NDC convention to pixels
    format = cam_dict['intrinsics_format']
    if format.lower() == "ndc_norm_image_bounds":
        # this is e.g. currently used in CO3D for storing intrinsics
        rescale = half_image_size_wh_orig
    elif format.lower() == "ndc_isotropic":
        rescale = half_image_size_wh_orig.min()
    else:
        raise ValueError(f"Unknown intrinsics format: {format}")

    # principal point and focal length in pixels
    principal_point_px = half_image_size_wh_orig - principal_point * rescale
    focal_length_px = focal_length * rescale

    # We need this in pixel so no need to do this here
    # # now, convert from pixels to PyTorch3D v0.5+ NDC convention
    # out_size = list(reversed(image_size))

    # half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
    # half_min_image_size_output = half_image_size_output.min()

    # # rescaled principal point and focal length in ndc 
    # principal_point = (
    #     half_image_size_output - principal_point_px * scale
    # ) / half_min_image_size_output
    # focal_length = focal_length_px * scale / half_min_image_size_output

    # TODO: This must be translatet to Extrinsics usable by Open3D. But I do not know how except installing pytorch3d
    # Initialise PerspectiveCameras(
        #     focal_length=focal_length[None],
        #     principal_point=principal_point[None],
        #     R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
        #     T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
        # )
    # And call get_world_to_view_transform which returns a 3D transform from world coordinates to the camera view coordinates (R, T) 
    return (focal_length_px, principal_point_px,
            torch.tensor(cam_dict['R'], dtype=torch.float), # Rotation Parameter (3x3)
            torch.tensor(cam_dict['T'], dtype=torch.float), # Translation Parameter (3)
            )


class DynamicReplica(SyntheticDepthDataset):
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
        self.max_depth = 65_504 # m ; As the Maximum 16 bit can hold.   
        self.min_depth = 1e-5 # m ; Value from official repo
        self.flip_to_open3d_coord = np.diag([1., 1., 1., 1.]) # TODO
        self.Cam_to_World = True # This means extrinsics are Cam_to_World

        # Load Paths. 
        # Dynamic replica comes with an json file which contains all paths and informations for the 
        # complete dataset. Since we just need the depth we extract only this, to be more efficient. 
        if self.is_test:
            split = 'test'
        if self.is_val:
            split = 'valid'
        else:
            split = 'train'
        
        frame_annotation_file = f'/export/scratch/ffeiden/data/dynamicReplica/{split}/frame_annotations_{split}.jgz'
        with gzip.open(frame_annotation_file, "rt", encoding="utf8") as zipfile:
            frame_annots_list = json.load(zipfile)
        
        # We iterate throug all paths. This could be more eficiant, since it looks like every scene is 
        # 300 frames long. But I did not test the test split regarding this. 
        sequence_name = ''
        camera_id = ''
        scene_dic = None

        for i in range(len(frame_annots_list)):
            if (sequence_name == frame_annots_list[i]['sequence_name']) & \
                (camera_id == frame_annots_list[i]['camera_name']):
                # As long as scene and camera doesent change we are in the same scene and 
                # can just append data 
                scene_dic['image'].append(os.path.join(root, split, frame_annots_list[i]['image']['path']))
                scene_dic['image_size'].append(frame_annots_list[i]['image']['size'])

                scene_dic['depth'].append(os.path.join(root, split, frame_annots_list[i]['depth']['path']))
                scene_dic['depth_scale'].append(frame_annots_list[i]['depth']['scale_adjustment'])
                scene_dic['valid_depth'].append(os.path.join(root, split, frame_annots_list[i]['depth']['mask_path']))

                scene_dic['cam'].append(frame_annots_list[i]['viewpoint'])
            else: 
                # If we enter a new scene we create a new scene list and save the old one
                if i != 0:
                    self.sample_list.append(scene_dic)
                scene_dic = {}
                sequence_name = frame_annots_list[i]['sequence_name']
                camera_id = frame_annots_list[i]['camera_name']

                scene_dic['image'] = [os.path.join(root, split, frame_annots_list[i]['image']['path'])]
                scene_dic['image_size'] = [frame_annots_list[i]['image']['size']]

                scene_dic['depth'] = [os.path.join(root, split, frame_annots_list[i]['depth']['path'])]
                scene_dic['depth_scale'] = [frame_annots_list[i]['depth']['scale_adjustment']]
                scene_dic['valid_depth'] = [os.path.join(root, split, frame_annots_list[i]['depth']['mask_path'])]

                scene_dic['cam'] = [frame_annots_list[i]['viewpoint']]
        self.sample_list.append(scene_dic) # Add also last scene

    def __extract_index_dash__(self, path):
        # Needs to be adapted for DynamicReplica
        base = path.split('.')[0]
        idx_str = base.split('-')[-1]
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
        num_frames = len(sample_paths['image'])
        h, w, c = cv2.imread(sample_paths['image'][0]).shape
        # prepare torch tensor for loading. 
        image = torch.zeros((num_frames, c, h, w))
        depth = torch.zeros((num_frames, h, w))
        valid_depth = torch.zeros_like(depth).to(dtype=torch.bool)
        depth_mask = torch.zeros_like(depth).to(dtype=torch.bool)
        Intrinsics = torch.zeros((num_frames, 3, 3))
        Extrinsics = torch.zeros((num_frames, 4, 4))
        
        for idx, image_path in enumerate(sample_paths['image']):
            # Load image 
            image_tmp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype('float32') / 255. # Directly convert in range 
            image[idx] = torch.from_numpy(rearrange(image_tmp, 'h w c -> c h w'))
            if self.verbose:
                img_idx = self.__extract_index_dash__(image_path)
                depth_idx = self.__extract_index__(sample_paths['depth'][idx])
                assert depth_idx == img_idx, f'Depth and Image Paths do not fit. Image: {image_path} Depth: {sample_paths['depth'][idx]}'
            
            # Load metric Depth
            depth_tmp = _load_16big_png_depth(sample_paths['depth'][idx]) # Depth is already in m: https://sintel-depth.csail.mit.edu/faq
            depth[idx] = torch.from_numpy(depth_tmp)

            # TODO: Check max & min values. Also load mask out of path an compare / return this as valid 
            valid_depth[idx] = torch.where((depth[idx] > self.min_depth) & (depth[idx] < self.max_depth), True, False)
            depth_mask_tmp = cv2.imread(sample_paths['valid_depth'][idx]).astype('float32') / 255. # Directly convert in range 
            depth_mask[idx] = torch.from_numpy(rearrange(depth_mask_tmp,'h w c -> c h w')).mean(dim=0)



            # Load Camera Pareters 
            focal_length, pricip_point, Rot, Trans = _get_pytorch3d_camera(cam_dict=sample_paths['cam'][idx], 
                                                                           image_size=sample_paths['image_size'][idx],
                                                                           scale=sample_paths['depth_scale'][idx])
            Intrinsics_tmp = torch.diag(torch.tensor([focal_length[0], focal_length[1], 1.]))
            Intrinsics_tmp[:2, 2] = pricip_point
            Intrinsics[idx] = Intrinsics_tmp
            Rot_tmp = torch.diag(torch.tensor([1., 1., 1., 1.]))
            Trans_tmp = torch.diag(torch.tensor([0., 0., 0., 0.]))
            Rot_tmp[:3, :3] = Rot
            Trans_tmp[:3, 3] = Trans 
            Extrinsics[idx] = Trans_tmp + Rot_tmp
        
        # Return dictionary
        sample_data = {}
        sample_data['image'] = image
        sample_data['depth'] = depth 
        sample_data['valid_depth'] = valid_depth
        sample_data['depth_mask'] = depth_mask
        sample_data['intrinsics'] = Intrinsics
        sample_data['extrinsics'] = Extrinsics

        return sample_data