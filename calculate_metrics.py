import argparse
import numpy as np
import os
import torch
import tifffile as tiff

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video
from torchinfo import summary
from torchvision.transforms import Compose
from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
from thop import profile
import torch.profiler as profiler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from einops import rearrange
import imageio
from natsort import natsorted
import subprocess
import warnings
import re
from utils.dc_utils import read_video_frames
from utils.util import compute_scale_and_shift
from utils.align import DepthMap, frame_align_lstsq
from utils.vis_util import visualise_data
from video_depth_anything.video_depth import INFER_LEN

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, default='/export/data/ffeiden/data/vkitti_videos')
parser.add_argument('--cam', type=str, default='Camera_0')
parser.add_argument('--device', type=str)
parser.add_argument('--encoder', type=str, choices=['vits', 'vitl'], default='vits')
parser.add_argument('--inference_length', type=int, default=32)
parser.add_argument('--fps', type=int, default=25)
parser.add_argument('--keyframe_list', type=str, nargs='+', default=['0', '12'])
parser.add_argument('--dont_generate', action='store_true')
parser.add_argument('--align_each_new_frame', action='store_true')
parser.add_argument('--add_gt_stab_line', action='store_true', help='Stores an extra image with the gt Stability line image')
parser.add_argument('--Scenes', nargs='+', default=['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20'], 
                    help='List of Scenes must be given as stings seperated with space')
parser.add_argument('--skip_tmp_block', action='store_true', help='Skips temporal block (second low dimension) for \
                    single image generation')

args = parser.parse_args()

# Argumente entpacken
root_dir = args.root_dir
cam = args.cam
device = args.device
encoder = args.encoder
context_length = args.inference_length
FPS = args.fps
keyframes = args.keyframe_list
generate = not args.dont_generate
align = args.align_each_new_frame
save_gt_stab_line = args.add_gt_stab_line
SCENE = args.Scenes
skip_tmp_block = args.skip_tmp_block

# Start running all. 
keyframes_name = ''
for key_num in keyframes:
    keyframes_name += str(key_num) + ','

if align:
    align_name = '_align'
else:
    align_name = ''
if skip_tmp_block:
    skip_name = 'skiped'
else:
    skip_name = ''

Name = f'New_SingleImage_{encoder}_con_{context_length}_{align_name}_keyframes_{keyframes}_{skip_name}'
if os.path.exists(os.path.join(root_dir, Name)):
    gen_root_dir = os.path.join(root_dir, Name)
    if generate:
        warnings.warn('WARNING: You are about to overwrite allready generated Results.')
else:
    os.mkdir(os.path.join(root_dir, Name))
    gen_root_dir = os.path.join(root_dir, Name)

vis_methods = ['VDA_s_vits', 'VDA_vits', 'DepthAny2_raw', 'DepthAny2']

gt_all_data = os.listdir(os.path.join(root_dir, 'gt_vids'))

# Sort Data
methods = ['DepthAny', 'DepthAny2', 'PrimeDepth']
path_dic = {}
path_dic['gt'] = natsorted([os.path.join(root_dir, 'gt_vids', p) for p in gt_all_data if cam+'_gt'+'.tiff' in p])
for key in methods:
    path_dic[key] = natsorted([os.path.join(root_dir, 'gt_vids', p) for p in gt_all_data if key+'_' in p and cam+'.tiff' in p])
    path_dic[key+'_raw'] = natsorted([os.path.join(root_dir, 'gt_vids', p) for p in gt_all_data if key+'_' in p and cam+'_raw.tiff' in p])

path_dic['rgb'] = [p.replace('_gt.tiff', '.mp4') for p in path_dic['gt']]

_, HEIGHT, WIDTH = tiff.imread(path_dic['gt'][0]).shape

if generate:
    cmd = ["python", "run.py", "--device", device, 
            "--output_dir", gen_root_dir,
            "--save_tiff", 
            "--save_vis",
            "--save_stats",
            "--encoder", encoder,
            "--keyframe_list", *keyframes,
            "--process_single_image",
            "--inference_length", str(context_length)]
    # Generate DepthAnythingVideo predictions with supprocess and save them in data 
    for video_path in path_dic['rgb']:
        cmd.append("--input_video")
        cmd.append(os.path.join(root_dir, 'gt_vids', video_path))
        if align:
            cmd.append("--align_each_new_frame")
        if skip_tmp_block:
            cmd.append("--skip_tmp_block")

        # run single image
        subprocess.run(cmd)
        
        # run original
        subprocess.run(["python", "run.py", "--device", device, 
                        "--input_video", os.path.join(root_dir, 'gt_vids', video_path),
                        "--output_dir", gen_root_dir,
                        "--save_tiff", 
                        "--save_vis",
                        "--save_stats",
                        "--encoder", encoder,])

# Update Paths:
gen_all_data = os.listdir(gen_root_dir)

path_dic[f'VDA_s_{encoder}'] = natsorted([os.path.join(gen_root_dir, p) for p in gen_all_data if 'Single_VideoDepthAny_' in p and '_depths.tiff' in p])
if len(path_dic[f'VDA_s_{encoder}']) == 0:
    warnings.warn('No VideoDepthAny with single image processing found, removed from list', UserWarning)
    del path_dic[f'VDA_s_{encoder}']

path_dic[f'VDA_{encoder}'] = natsorted([os.path.join(gen_root_dir, p) for p in gen_all_data if 'VideoDepthAny_' in p and '_depths.tiff' in p and 'Single_' not in p])
if len(path_dic[f'VDA_{encoder}']) == 0:
    warnings.warn('No VideoDepthAny with single image processing found, removed from list', UserWarning)
    del path_dic[f'VDA_{encoder}']

def resize(frame_list):
    return np.array([cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA) for frame in frame_list])

# Load data
data_dic = {}
for key in tqdm(path_dic):
    if key != 'rgb':
        for p in path_dic[key]:
            scene_pattern = re.compile(r"(Scene\d+)")
            scene = scene_pattern.search(p).group(1)
            data = np.array(tiff.imread(p))
            _, height, width = data.shape
            if height != HEIGHT or width != WIDTH:
                data = resize(data)
            data_dic[key+f'_{scene}'] = data
        
    else:
        for p in path_dic[key]:
            scene_pattern = re.compile(r"(Scene\d+)")
            scene = scene_pattern.search(p).group(1)
            data, _ = read_video_frames(p, process_length=-1, target_fps=-1, max_res=-1)
            _, height, width, _ = data.shape
            if height != HEIGHT or width != WIDTH:
                data = resize(data)
            data_dic[key+f'_{scene}'] = data

for key in data_dic: 
    if '_raw' in key or 'VDA_' in key:
        scene_pattern = re.compile(r"(Scene\d+)")
        scene = scene_pattern.search(key).group(1)
        # We only align on the first frame for now!
        if 'VDA_s_' in key:
            prediction = data_dic[key][0]
            warm_up = len(data_dic[f'gt_{scene}']) - len(data_dic[key])
            ground_truth = data_dic[f'gt_{scene}'][warm_up]
        else:
            prediction = data_dic[key][0]
            ground_truth = data_dic[f'gt_{scene}'][0]
        
        valid_depth = ground_truth < 80.
        gt_mask = np.ma.array(ground_truth, mask=~valid_depth)
        prediction_mask = np.ma.array(prediction)
        # Raw predictions (have to be) are always inverse depth
        prediction_tmp = DepthMap(prediction_mask, inverse=True, range=None, scale=None, shift=None)
        gt_depth_tmp = DepthMap(gt_mask, inverse=False, range=None, scale=1, shift=0)
        # Calculate scale & shift for INVERSE depth
        alignment = frame_align_lstsq(prediction_tmp, gt_depth_tmp)
        scale, shift = alignment.scale, alignment.shift
        
        # Use scale & shift to align --> This is still inverse depth here!
        data_dic[key] = np.clip((data_dic[key] - shift) / scale, 0., 1.)
        
        # To make it metric we need to invert it again.
        # Avoid division by 0
        data_dic[key] = np.where( data_dic[key] == 0., 1e-4, data_dic[key]) 
        # Clip to max depth 
        data_dic[key] = np.clip(1. / data_dic[key], 0., 80.)

# Reopen logging file
log_file = os.path.join(gen_root_dir, 'inference_log.txt')
log_lines = []

Metrics_dic = {}
for method in vis_methods:
    Metrics_dic[method+'_MSE'] = []
    Metrics_dic[method+'_Abs'] = []

for scene in SCENE:
    gt_depth_vid = data_dic['gt_'+scene]
    for method in vis_methods:
        method_pred = data_dic[method + '_' + scene]
        try:
            error = np.abs(gt_depth_vid - method_pred).sum() / gt_depth_vid.size
            MSE_err = np.mean((method_pred - gt_depth_vid)**2)
        except ValueError:
            warm_length = len(gt_depth_vid) - len(method_pred)
            MSE_err = np.mean((method_pred - gt_depth_vid[warm_length:])**2)
            log_lines.append(f'Assuming wamup length of {warm_length}')
            error = np.abs(gt_depth_vid[warm_length:] - method_pred).sum() / gt_depth_vid.size
        
        log_lines.append(f'{scene} {method}: ')
        log_lines.append('-----------------------')
        log_lines.append(f'Abs. Error: {error:.3f}')
        log_lines.append(f'MSE Error: {MSE_err:.3f}')
        log_lines.append('')
        Metrics_dic[method+'_MSE'].append(MSE_err)
        Metrics_dic[method+'_Abs'].append(error)

for method in vis_methods:
    error = np.mean(Metrics_dic[method+'_Abs'])
    MSE_err = np.mean(Metrics_dic[method+'_MSE'])


    log_lines.append(f'Overall {method}: ')
    log_lines.append('-----------------------')
    log_lines.append(f'Abs. Error: {error:.3f}')
    log_lines.append(f'MSE Error: {MSE_err:.3f}')
    log_lines.append('')

with open(log_file, "a") as f:
    for line in log_lines:
        f.write(line + "\n")

visualise_data(SCENE=SCENE, FPS=FPS, HEIGHT=HEIGHT, WIDTH=WIDTH, 
               data_dic=data_dic, root=gen_root_dir, methods=vis_methods, scene_idx=0, save_gt_stability=save_gt_stab_line)

visualise_data(SCENE=SCENE, FPS=FPS, HEIGHT=HEIGHT, WIDTH=WIDTH, 
               data_dic=data_dic, root=gen_root_dir, methods=vis_methods, scene_idx=1)

visualise_data(SCENE=SCENE, FPS=FPS, HEIGHT=HEIGHT, WIDTH=WIDTH, 
               data_dic=data_dic, root=gen_root_dir, methods=vis_methods, scene_idx=2)

visualise_data(SCENE=SCENE, FPS=FPS, HEIGHT=HEIGHT, WIDTH=WIDTH, 
               data_dic=data_dic, root=gen_root_dir, methods=vis_methods, scene_idx=3)

visualise_data(SCENE=SCENE, FPS=FPS, HEIGHT=HEIGHT, WIDTH=WIDTH, 
               data_dic=data_dic, root=gen_root_dir, methods=vis_methods, scene_idx=4)

