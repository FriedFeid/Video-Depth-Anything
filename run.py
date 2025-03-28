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
import argparse
import numpy as np
import os
import torch
import tifffile as tiff

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video
from torchinfo import summary
import time
import psutil
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--device', type=str, help='Device Name in form cuda:0')
    parser.add_argument('--input_video', type=str, default='/export/data/ffeiden/data/vkitti_videos/gt_vids/Scene01_clone_Camera_0.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--process_single_image', action='store_true', help='Only process individual Images instead of batches of 32')

    # TODO: Implement these flags
    parser.add_argument('--inference_length', type=int, default=32, help='The total amount of context frames given to the motion module.\
                        This includes keyframes')
    parser.add_argument('--keyframe_list', type=int, nargs='+', default=[0, 12], help='List of keyframes used. The first one must be 0.\
                        The following index gives the position in the inference_length batch. This means e.g.: if second index is 12 and \
                        inferenz_legth=32 the keyframe is 42 frames before the current frame. \
                        ((inference_length) 32 - (len(keyfram_list) 2) + (value keyframe_list) 12). Default "0 12" ')
    parser.add_argument('--align_each_new_frame', action='store_true', help='If set it will for each frame predicted use the keyframe_list\
                         to calculate scale & shift of the current forward (forwards all keyframes) and uses the scale & shift to aling \
                        new frame.')
    parser.add_argument('--original', action='store_true', help='Runns the original model with no adjustments. WARINING: Overwrites\
                        --process_single_image, --inferenz_lenght, --keyframe_list, align_each_new_frame')
    parser.add_argument('--skip_tmp_block', action='store_true', help='Skips second Temporal Block')
    
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_tiff', action='store_true', help='save as tiff image stack')
    parser.add_argument('--save_orig', action='store_true', help='saves the Original video as well')
    parser.add_argument('--save_vis', action='store_true', help='saves a visualisation of the Video')
    parser.add_argument('--save_stats', action='store_true', help='Saves out the timing and memory consumpution. As well as other stats')

    args = parser.parse_args()
    assert max(args.keyframe_list) < args.inference_length # Adjust code to make this work as well (to use smaler widnow )
    #TODO: sort keyframe_list to make it compatible with the code. 
    DEVICE = args.device if torch.cuda.is_available() else 'cpu'

    # Setup Logging File
    if args.output_dir != './outputs':
        log_file = os.path.join(args.output_dir, 'inference_log.txt')
    else:
        log_file = 'inference_log.txt'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    # Reset Memory recording:
    if args.save_stats:
        torch.cuda.reset_peak_memory_stats(DEVICE)
        process = psutil.Process(os.getpid())

    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    total_images = len(frames)
    # frames: [447, 374, 1242, 3] [frames, height, width, channels] in range 0, 255 type uint8
    # target_fps: float
    if args.save_stats:
        start_time = time.time()
    if args.process_single_image:
        depths, fps = video_depth_anything.infere_single_image(frames, target_fps, device=DEVICE, fp32=args.fp32, input_size=args.input_size,
                                                               inference_length=args.inference_length, keyframe_list=args.keyframe_list,
                                                               align_each_new_frame=args.align_each_new_frame, warmup=True, skip_tmp_block=args.skip_tmp_block)
    elif args.original:
        depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
    else:
        depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32, skip_tmp_block=args.skip_tmp_block)
    # Summarize stats of the run
    if args.save_stats:
        end_time = time.time()
        duration = end_time - start_time
        processed_frames = len(depths)
        process_fps = processed_frames / duration
        total_fps = total_images / duration
        # Ressource usage
        gpu_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024**2  # in MB
        ram_mb = process.memory_info().rss / 1024**2  # in MB
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if args.process_single_image:
            single_name = 'Single Image'
        else:
            single_name = ''
        if args.align_each_new_frame:
            align_name = 'Align each frame'
        else:
            align_name = ''

        log_lines = [
            f"Run: {args.encoder} {args.inference_length} {single_name} {align_name} \
             keyframes {args.keyframe_list} {timestamp}",
            "",
            "Times",
            f"___________________________",
            f"Runtime: {duration}",
            f"Processed Frames: {processed_frames}",
            f"Total Frames: {total_images}",
            f"FPS (Processed): {process_fps}",
            f"Raw FPS: {total_fps}",
            "",
            "Memory",
            f"___________________________",
            f"GPU Memory (mb): {gpu_mem_mb}",
            f"RAM Memory (mb): {ram_mb}",
            "",
            "",
        ]
        with open(log_file, "a") as f:
            for line in log_lines:
                f.write(line + "\n")

    
    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.process_single_image:
        name = f'Single_VideoDepthAny_{args.encoder}_'+os.path.splitext(video_name)[0]
    else:
        name = f'VideoDepthAny_{args.encoder}_'+os.path.splitext(video_name)[0]

    processed_video_path = os.path.join(args.output_dir, name+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, name+'_vis.mp4')
    if args.save_orig:
        save_video(frames, processed_video_path, fps=fps)
    if args.save_vis:
        save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale, spectral=True if not args.grayscale else False)

    if args.save_npz:
        depth_npz_path = os.path.join(args.output_dir, name+'_depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
    if args.save_tiff:
        tiff.imwrite(os.path.join(args.output_dir, name+'_depths.tiff'), depths, photometric='rgb')

    


