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
from .dpt import DPTHead
from .motion_module.motion_module import TemporalModule
from easydict import EasyDict


class DPTHeadTemporal(DPTHead):
    def __init__(self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken)

        assert num_frames > 0
        motion_module_kwargs = EasyDict(num_attention_heads                = 8,
                                        num_transformer_block              = 1,
                                        num_attention_blocks               = 2,
                                        temporal_max_len                   = num_frames,
                                        zero_initialize                    = True,
                                        pos_embedding_type                 = pe)

        self.motion_modules = nn.ModuleList([
            TemporalModule(in_channels=out_channels[2], 
                           **motion_module_kwargs),
            TemporalModule(in_channels=out_channels[3],
                           **motion_module_kwargs),
            TemporalModule(in_channels=features,
                           **motion_module_kwargs),
            TemporalModule(in_channels=features,
                           **motion_module_kwargs)
        ])

    def forward(self, out_features, patch_h, patch_w, frame_length):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            B, T = x.shape[0] // frame_length, frame_length
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        B, T = layer_1.shape[0] // frame_length, frame_length

        layer_3 = self.motion_modules[0](layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        layer_4 = self.motion_modules[1](layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4 = self.motion_modules[2](path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3 = self.motion_modules[3](path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True  # What happens if we here interpolate with nearest instead of bilinear  ? might help edges ? 
        )
        ori_type = out.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.scratch.output_conv2(out.float())

        return out.to(ori_type)
    
    def get_motion_features(self, out_features, patch_h, patch_w):
        '''
        :param out_features: Extracted featuers out of the DinoV2 Backbone in shape: [1, patch_size, embeddings]
        :type out_features: torch.Tensor
        :param patch_h: height of the batch
        :type patch_h: int
        :param patch_w: width of the batch
        :type patch_w: int
        :return:
                layer_3 (torch.Tensor) [batch, embed, height, width]
                layer_4 (torch.Tensor) [batch, embed, height, width]
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        '''
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        _, _, layer_3, layer_4 = out
        return layer_3, layer_4
    
    def foward_single_image(self, out_features, patch_h, patch_w, frame_length, motion_features):
        '''
        :param out_features: Extracted featuers out of the DinoV2 Backbone in shape: [1, patch_size, embeddings]
        :type out_features: torch.Tensor
        :param patch_h: height of the batch
        :type patch_h: int
        :param patch_w: width of the batch
        :type patch_w: int
        :param frame_length: how many frames are computet per batch 
        :type frame_length: int
        :param motion_features: Tuple of torch.Tensor with the layer_3 and layer_4 as input for the attention
                                Shape [1, patch_size, embeddings]
        :type motion_features: Tuple[torch.Tensor, torch.Tensor]
        
        :return: 
            - out (torch.Tensor): Estimation of new image.

            - layer_3 (torch.Tensor): New layer_3 with new frame.

            - layer_4 (torch.Tensor): New layer_4 with new frame.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        '''
        
        # First do everything that can be done with batch size of 1
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            B, T = x.shape[0] // frame_length, frame_length
            if B == 0: 
                B = 1
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)

        # Do computation for multible Frames

        # Move input one frame further
        layer_3_old, layer_4_old = motion_features
        layer_3 = torch.cat([layer_3_old[1:], layer_3], dim=0)
        layer_4 = torch.cat([layer_4_old[1:], layer_4], dim=0)

        # Smallest resolution F4
        layer_4 = self.motion_modules[1](layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # Normal resolution F3
        layer_3 = self.motion_modules[0](layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        layer_3_rn = self.scratch.layer3_rn(layer_3)

        # Upsampling to resolution F3
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        # Motion Handeling F4 in resolution F3
        path_4 = self.motion_modules[2](path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)

        # Adding F3 to F4 and upsampling to resolution of F2
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        # Motion Handeling of F3&F4 in resolution of F2
        path_3 = self.motion_modules[3](path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)


        # Compute the rest in single batch fassion
        path_3 = path_3[-1].unsqueeze(0)
        # Adding F2 to F3 and resizing to resolution F1
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        # Adding F2 to F1
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # Output
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True  # What happens if we here interpolate with nearest instead of bilinear  ? might help edges ? 
        )
        ori_type = out.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.scratch.output_conv2(out.float())

        return out.to(ori_type), layer_3, layer_4

        