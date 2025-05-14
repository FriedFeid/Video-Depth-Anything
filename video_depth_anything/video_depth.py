# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)

    def forward(self, x, skip_tmp_block=False):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.head(features, patch_h, patch_w, T, skip_tmp_block)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]
    
    def forward_single_image(self, x, motion_features, pred_depth_idx=None, inference_length=32, skip_tmp_block=False):
        '''
        :param x: Image of size [1, 1, 3, height, width]
        :type x: torch.Tensor
        :param motion_features: All features of the DinoV2 Encoder used to compute next batch
        :type motion_features: Tuple(torch.Tensors, torch.Tensor, torch.Tensor, torch.Tensor)
        :param pred_depth_idx: Indexes for wich features a depth should be predicted. The rest is only used for
                               Motion feature computation
        :type pred_depth_idx: List[int]
        '''
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)

        single_depth, layer_1, layer_2, layer_3, layer_4 = self.head.foward_single_image(features, patch_h=patch_h, patch_w=patch_w, frame_length=inference_length,
                                                                       motion_features=motion_features, pred_depth_idx=pred_depth_idx, skip_tmp_block=skip_tmp_block)
        single_depth = F.interpolate(single_depth, size=(H, W), mode="bilinear", align_corners=True)
        single_depth = F.relu(single_depth)
        motion_features = (layer_1, layer_2, layer_3, layer_4)
        if pred_depth_idx is None:
            return single_depth.squeeze(1).unflatten(0, (B, T)), motion_features
        else:
            return single_depth.squeeze(1).unflatten(0, (B, len(pred_depth_idx) + 1)), motion_features # +1 because of the current prediction

    def infere_single_image(self, frames, target_fps, input_size=518, device='cuda', fp32=False, warmup=True,
                            inference_length=32, keyframe_list=[0,12],
                            align_each_new_frame=True, skip_tmp_block=False):
        '''
        :param frames: List of all frames in the Video
        :type frames: List
        :param target_fps: Number of fps for the final video
        :type target_fps: int
        :param input_size: Input size to calculate the ratio for more efficient processing
        :type input_size: int
        :param device: device to put data on (same as model)
        :type device: str
        :param fp32: Specifises if floating point 32 are used or floating point 16
        :type fp32: bool
        :param warmup: If set to true the first prediction will be done after 32 frames. 
                       Otherwise it will predict the first 32 frames in the "normal" mode
        :type warmup: bool
        :param inference_length: Number of reference frames for the motion module. 
        :type inference_length: int
        :param keyframe_list: List of keyframes for the motion module (frames which are not in temporal order)
        :type keyframe_list: List[int]
        :param align_each_new_frame: If set to true each new frame will be aligned using scale and shift calculatet with the keyframes of the batch
        :type align_each_new_frame: bool 

        :return:
            - depth (nd.Array): The estimated depth of each frame (as a batch)

            - FPS (int): FPS of the video
        :rtype: Tuple[nd.Array, int]
        '''

        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14
        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        # Calculate length of appended frame list and extend
        frame_list = [frames[i] for i in range(frames.shape[0])]
        org_video_len = len(frame_list)

        depth_list = []

        # prepare for keyframe saving and propagation
        max_keyframe = max(keyframe_list)
        # How to shift the saved features to dont waste memory
        move_motion_features = [i for i in range(inference_length+max_keyframe-1) if i != 1]
        # Distance to current frame idx
        distance_to_batch = [keyframe_list[idx] + (inference_length-len(keyframe_list)) for idx in range(len(keyframe_list))]

        # Now get the keyframe idx for wich we predict the next depth

        # Start by checking if keyframes are within the Context interval or not. 
        static_keyframes = []
        for idx in range(len(keyframe_list)):
            if inference_length > keyframe_list[idx]:
                static_keyframes.append(inference_length - keyframe_list[idx])
            elif inference_length <= keyframe_list[idx]:
                static_keyframes.append(idx+1)
        assert not len(static_keyframes) != len(set(static_keyframes)), f'Setup leads to duplicates in the keyframes: {static_keyframes}'

        # Calculate The index list for the starting frames
        use_feature_idx = []
        if align_each_new_frame:
            Align_idx = [] # Relative idx in prediction batch to use as alignment frame
        for frame_idx in range(inference_length-1, inference_length+max_keyframe, 1):

            tmp_batch_idx = [idx for idx in range(frame_idx-(inference_length-1), frame_idx, 1)]
            tmp_batch_idx[0] = 0 # We always use first frame as reference
            if align_each_new_frame:
                align_idx_batch = [0]
            
            for idx, static_keyframe in enumerate(static_keyframes):
                if static_keyframe in tmp_batch_idx:
                    if align_each_new_frame:
                        align_idx_batch.append(tmp_batch_idx.index(static_keyframe))
                else:
                    if align_each_new_frame:
                        align_idx_batch.append(idx+1)
                    
                    if frame_idx-distance_to_batch[idx] <= static_keyframe:
                        tmp_batch_idx[idx+1] = static_keyframe
                    else:
                        tmp_batch_idx[idx+1] = static_keyframe + (frame_idx - distance_to_batch[idx] - static_keyframe)
            
            use_feature_idx.append(tmp_batch_idx)
            if align_each_new_frame:
                Align_idx.append(align_idx_batch)
        
        # Prepare variables used in the loop 
        features_f1 = None
        features_f2 = None
        features_f3 = None
        features_f4 = None

        depth_pred_idx = None
        old_keyframes = None
        if align_each_new_frame:
            abs_pred_depth_idx = None
        
        if warmup:
            for i in tqdm(range(len(frame_list))):
                cur_frame = torch.from_numpy(transform({'image': frame_list[i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0).to(device)
                
                if i < inference_length-1:
                    # Accumulate all motion features we need for the first prediction
                    B, F_, C, H, W = cur_frame.shape
                    patch_h, patch_w = H // 14, W // 14
                    with torch.no_grad():
                        with torch.autocast(device_type=device, enabled=(not fp32)):
                            features = self.pretrained.get_intermediate_layers(cur_frame.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
                            layer_1_tmp, layer_2_tmp, layer_3_tmp, layer_4_tmp = self.head.get_motion_features(features, patch_h, patch_w)

                    if i == 0:
                        # Prepare torch tensors for saving. We want to use as less memory as possible. So we reserve the max length with 0
                        features_f1 = torch.zeros(inference_length+max_keyframe-1, *layer_1_tmp.size()[1:]).to(device)
                        features_f2 = torch.zeros(inference_length+max_keyframe-1, *layer_2_tmp.size()[1:]).to(device)
                        features_f3 = torch.zeros(inference_length+max_keyframe-1, *layer_3_tmp.size()[1:]).to(device)
                        features_f4 = torch.zeros(inference_length+max_keyframe-1, *layer_4_tmp.size()[1:]).to(device)

                        # Fill in first Features
                        features_f1[0] = layer_1_tmp[0]
                        features_f2[0] = layer_2_tmp[0]
                        features_f3[0] = layer_3_tmp[0]
                        features_f4[0] = layer_4_tmp[0]
                    
                    else:
                        features_f1[i] = layer_1_tmp[0]
                        features_f2[i] = layer_2_tmp[0]
                        features_f3[i] = layer_3_tmp[0]
                        features_f4[i] = layer_4_tmp[0]
                    
                    # Start actual predictions. 
                else:
                    if i < inference_length+max_keyframe:
                        # Use the changing feature list. 
                        predict_f1 = features_f1[use_feature_idx[i-(inference_length-1)]]
                        predict_f2 = features_f2[use_feature_idx[i-(inference_length-1)]]
                        predict_f3 = features_f3[use_feature_idx[i-(inference_length-1)]]
                        predict_f4 = features_f4[use_feature_idx[i-(inference_length-1)]]
                        if align_each_new_frame:
                            # Since the batch is not moved jet, the absolute frame is equal with the use_feature_idx entry of the keyframe. 
                            abs_pred_depth_idx = [use_feature_idx[i-(inference_length-1)][tmp_idx] for tmp_idx in Align_idx[i-(inference_length-1)]]
                            if i == inference_length-1:
                                # We predict the first batch all Depths we need.
                                depth_pred_idx = use_feature_idx[i-(inference_length-1)]
                            else:
                                depth_pred_idx = Align_idx[i-(inference_length-1)]
                            
                    else:
                        # Use the last entry, since nothing is changing. We now shift the features_f1 ... itself.
                        predict_f1 = features_f1[use_feature_idx[-1]]
                        predict_f2 = features_f2[use_feature_idx[-1]]
                        predict_f3 = features_f3[use_feature_idx[-1]]
                        predict_f4 = features_f4[use_feature_idx[-1]]
                        if align_each_new_frame:
                            tmp_abs_pred_depth_idx = [use_feature_idx[-1][tmp_idx] for tmp_idx in Align_idx[-1]]
                            depth_pred_idx = Align_idx[-1]

                            abs_pred_depth_idx = []
                            for idx in tmp_abs_pred_depth_idx:
                                if idx == 0:
                                    abs_pred_depth_idx.append(0)
                                else:
                                    # Because when we are here we have moved the batch already once we need to add one 
                                    abs_pred_depth_idx.append(idx + (i - (inference_length + max_keyframe )) + 1)
                    
                    # Set up the motion features and predict Depth
                    motion_features = (predict_f1, predict_f2, predict_f3, predict_f4)
                    with torch.no_grad():
                        with torch.autocast(device_type=device, enabled=(not fp32)):
                            depth, motion_features = self.forward_single_image(cur_frame, motion_features, pred_depth_idx=depth_pred_idx, inference_length=inference_length,
                                                                               skip_tmp_block=skip_tmp_block)
                    
                    # Save new motion_features for the next step
                    new_feature_1, new_feature_2, new_feature_3, new_feature_4 = motion_features
                    
                    if i < inference_length+max_keyframe-1:
                        # As long as the maximum number has not reachted just fill in
                        features_f1[i] = new_feature_1[0]
                        features_f2[i] = new_feature_2[0]
                        features_f3[i] = new_feature_3[0]
                        features_f4[i] = new_feature_4[0]
                    else:
                        features_f1[:-1] = features_f1[move_motion_features]
                        features_f2[:-1] = features_f2[move_motion_features]
                        features_f3[:-1] = features_f3[move_motion_features]
                        features_f4[:-1] = features_f4[move_motion_features]

                        features_f1[-1] = new_feature_1[0]
                        features_f2[-1] = new_feature_2[0]
                        features_f3[-1] = new_feature_3[0]
                        features_f4[-1] = new_feature_4[0]
                    
                    depth = depth.to(cur_frame.dtype)
                    depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)

                    # Handle real time alignment
                    #TODO: check if frames are correctly picked & if alignment works correct
                    if align_each_new_frame:
                        if old_keyframes is None:
                            old_keyframes = []
                            depth_list += [depth[k][0].cpu().numpy() for k in range(depth.shape[0])]
                        else:
                            cur_prediction = depth[-1][0].cpu().numpy()
                            cur_keyframes = [depth[k][0].cpu().numpy() for k in range(len(depth_pred_idx))]

                            old_keyframes = [depth_list[j] for j in abs_pred_depth_idx]
                            scale, shift = compute_scale_and_shift(np.concatenate(cur_keyframes) if len(depth_pred_idx) > 1 else cur_keyframes,
                                                                   np.concatenate(old_keyframes) if len(abs_pred_depth_idx) > 1 else old_keyframes,
                                                                   np.concatenate(np.ones_like(old_keyframes)==1) if len(abs_pred_depth_idx) > 1 else np.ones_like(old_keyframes)==1)
                            cur_prediction = cur_prediction * scale + shift
                            
                            depth_list.append(cur_prediction)
                    else:
                        depth_list += [depth[k][0].cpu().numpy() for k in range(depth.shape[0])]
        
        else:
            raise NotImplementedError

        if align_each_new_frame:
            return np.stack(depth_list[1:org_video_len], axis=0), target_fps # Because the first frame is only used for alignment
        else:
            return np.stack(depth_list[:org_video_len], axis=0), target_fps
    
    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', fp32=False, skip_tmp_block=False):
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        # Calculate length of appended frame list and extend
        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len
        
        depth_list = []
        pre_input = None  # pre_input: [b, frames, channels, height, width] same as curr_input to calculate overlap
        for frame_id in tqdm(range(0, org_video_len, frame_step)): #frame_step 22 von 0 - 447
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)  # cur_input: [1, 32, 3, 280, 924], [batch_size, frames, channel, height, width]; min: -3.2, max: 3.1
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth = self.forward(cur_input, skip_tmp_block) # depth shape: [1, T (frames), H (height), W (width)], min: 0, max: 2390

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)  # Go back to input dimensions
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

            pre_input = cur_input

        del frame_list
        gc.collect()

        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id+i])
                scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                       np.concatenate(ref_align),
                                                       np.concatenate(np.ones_like(ref_align)==1))

                pre_depth_list = depth_list_aligned[-INTERP_LEN:] # Vorheriger last 8 for interpolation
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP] #Overlap frames out of predictions
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift #scale it to fit with previous
                    post_depth_list[i][post_depth_list[i]<0] = 0 # Clip negative
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                for i in range(OVERLAP, INFER_LEN): # scale 22 rest 22 frames and save as prediction
                    new_depth = depth_list[frame_id+i] * scale + shift
                    new_depth[new_depth<0] = 0
                    depth_list_aligned.append(new_depth)

                ref_align = ref_align[:1] # Cut out last aling frame (12)
                for kf_id in kf_align_list[1:]: # For the last aling frames (12)
                    new_depth = depth_list[frame_id+kf_id] * scale + shift # Take the 12 frame of current pred batch and scale and shift it 
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth) # Means 0 stays always the same frame. Always beginning of sequence!!
            
        depth_list = depth_list_aligned
            
        return np.stack(depth_list[:org_video_len], axis=0), target_fps
        