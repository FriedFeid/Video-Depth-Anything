# TODO: Get Extrinsics to combine Images of Kitti
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
from datasets.Kitti import KITTI
from einops import rearrange
from utils.metrics import OutlierRatio, SignedRelativeDifference_Error, AbsoluteRelativeDifference_Error, MeanSquared_Error, AbsoluteDifference_Error
from utils.metrics import Outlier, SignedRelativeDifference, AbsoluteDifference, MeanSquared, AbsoluteRelativeDifference
from utils.align import align_prediction
import matplotlib.pyplot as plt
from utils.metrics import csv_saver
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Video Depth Anything')

    # Must be given:
    parser.add_argument('--device', type=str, help='Device Name in form cuda:0')
    parser.add_argument('--output_dir', type=str, default='/export/scratch/ffeiden/VideoDepthPrediction/EvalKitti/')
    parser.add_argument('--dataset', type=str, default='Kitti', choices=['Kitti', 'Kitti_val'])

    # Network Inference Parameter
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--process_single_image', action='store_true', help='Only process individual Images instead of batches of 32')
    parser.add_argument('--inference_length', type=int, default=32, help='The total amount of context frames given to the motion module.\
                        This includes keyframes')
    parser.add_argument('--keyframe_list', type=int, nargs='+', default=[20], help='List of keyframes in addition to 0 used.\
                        The value of the keyframe gives the distance form the batch to the keyframe. \
                        ((inference_length) 32 - (len(keyfram_list) 2) + (value keyframe_list) 12). Default "12" ')
    parser.add_argument('--align_each_new_frame', action='store_true', help='If set it will for each frame predicted use the keyframe_list\
                         to calculate scale & shift of the current forward (forwards all keyframes) and uses the scale & shift to aling \
                        new frame.')
    parser.add_argument('--original', action='store_true', help='Runns the original model with no adjustments. WARINING: Overwrites\
                        --process_single_image, --inferenz_lenght, --keyframe_list, align_each_new_frame')
    parser.add_argument('--skip_tmp_block', action='store_true', help='Skips second Temporal Block')
    
    # Input handling of Network
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')

    # Alignment parameter
    parser.add_argument('--align_only_first_frame', action='store_true', help='If set to true it will save out results which are only aligned on\
                        first frame')

    args = parser.parse_args()

    assert args.inference_length > len(args.keyframe_list) + 2, 'Inference length to small for the number of geiven keyframes'

    # Initialise Network
    DEVICE = args.device if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
    # Reset Memory recording:
    torch.cuda.reset_peak_memory_stats(DEVICE)
    process = psutil.Process(os.getpid())

    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    if args.dataset == 'Kitti_val':
        dataset = KITTI(is_val=True)
        fps = 10
        target_fps = 10
    elif args.dataset == 'Kitti':
        dataset = KITTI()
        fps = 10
        target_fps = 10

    # Setup Output path:
    if args.process_single_image:
        single_name = 'Single Image'
    else:
        single_name = ''
    if args.align_each_new_frame:
        align_name = 'Align each frame'
    else:
        align_name = ''

    name = args.encoder+'_'+single_name+'_'+align_name+'_'+str(args.inference_length)

    # Set up Inference Timing & Memory stats
    FPS_processed = []
    GPU_mem = []
    CPU_mem = []
    Log_metrics = csv_saver(os.path.join(args.output_dir, f'{args.dataset}_{name}.csv'))
    if args.align_only_first_frame:
        Log_first_frame = csv_saver(os.path.join(args.output_dir, f'{args.dataset}_{name}_firstFrameAlign.csv'))

    
    # Start inference Loop over all Scenes
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        # Get Scene Name for documentation
        scene_name = dataset.sample_list[i]
        if 'Kitti' in args.dataset:
            scene_name = scene_name['image'][0].split('/')[8] + '_cam'+dataset.sample_list[i]['cameraID']
        else:
            KeyError('Could not extract Scenen Name, because Dataset is unknown')

        # Get Frames of Whole Scene
        frames = (rearrange(sample['image'].numpy(), 'b c h w -> b h w c') * 255).astype(np.uint8)

        # Infere Video
        total_images = len(frames)
        # Reset Memory stats
        torch.cuda.reset_peak_memory_stats(DEVICE)

        start_time = time.time()
        if args.process_single_image:
            if total_images > args.inference_length + 1: # If there are not enought frames for buffering, we can not compute this and need to skip
                depths, fps = video_depth_anything.infere_single_image(frames, target_fps, device=DEVICE, fp32=False, input_size=args.input_size,
                                                                       inference_length=args.inference_length, keyframe_list=args.keyframe_list,
                                                                       align_each_new_frame=args.align_each_new_frame, warmup=True, skip_tmp_block=args.skip_tmp_block)
            else:
                continue
        else:
            depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=False)

        end_time = time.time()
        duration = end_time - start_time
        processed_frames = len(depths)
        process_fps = processed_frames / duration
        total_fps = total_images / duration
        # Ressource usage
        gpu_mem_mb = torch.cuda.max_memory_reserved(DEVICE) / 1024**2  # in MB
        ram_mb = process.memory_info().rss / 1024**2  # in MB
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        FPS_processed.append(process_fps)
        GPU_mem.append(gpu_mem_mb)
        CPU_mem.append(ram_mb)
                
        # Estimate metrics and log them for each scene individually
        valid_depth = sample['valid_depth'].numpy()
        prediction = depths
        ground_truth = sample['depth'].numpy()

        if len(ground_truth) != len(prediction):
            warmup = len(ground_truth) - len(prediction)
            print(f'Assuming warm up Length of {warmup}')
            
            ground_truth = ground_truth[warmup:]
            valid_depth = valid_depth[warmup:]

        # Align on all frames here
        # TODO: For evaluation normaly you would mybe here clip valid depth to 80 meters. However we do not do this 
        align_pred, scale, shift = align_prediction(prediction=prediction, ground_truth=ground_truth, valid_depth=valid_depth, max_depth=dataset.max_depth)
        Log_metrics.save_metrics_csv(prediction=align_pred, ground_truth=ground_truth, scale=scale, shift=shift,
                                     scene_name=scene_name, valid_depth=valid_depth, frames=total_images)
        
        # Align on only first frame here
        if args.align_only_first_frame:
            # TODO: For evaluation normaly you would mybe here clip valid depth to 80 meters. However we do not do this 
            _, scale, shift = align_prediction(prediction=prediction[0], ground_truth=ground_truth[0], valid_depth=valid_depth[0], max_depth=dataset.max_depth)
            
            # Use scale & shift to align --> This is still inverse depth here!
            align_pred = np.clip((prediction - shift) / scale, 0., 1.)
            # To make it metric we need to invert it again.
            # Avoid division by 0
            align_pred = np.where( align_pred == 0., 1e-4, align_pred) 
            # Clip to max depth 
            align_pred = np.clip(1. / align_pred, 0., dataset.max_depth)

            Log_first_frame.save_metrics_csv(prediction=align_pred, ground_truth=ground_truth, scale=scale, shift=shift,
                                             scene_name=scene_name, valid_depth=valid_depth, frames=total_images)

    # Summarize metrics in csv file 
    Additional_Header = ['Mean FPS', 'Var FPS', 'Mean GPU (Mb)', 'Var GPU', 'Mean Ram (Mb)', 'Var Ram']
    Additional_Infos = [np.mean(FPS_processed), np.var(FPS_processed), 
                        np.mean(GPU_mem), np.var(GPU_mem), 
                        np.mean(CPU_mem), np.var(CPU_mem)]
    Log_metrics.summarize_metrics_csv(additional_infos_Header=Additional_Header, additional_infos_data=Additional_Infos)

    if args.align_only_first_frame:
        Log_first_frame.summarize_metrics_csv(additional_infos_Header=Additional_Header, additional_infos_data=Additional_Infos)

# python eval.py --device cuda:0 --dataset Kitti --process_single_image --align_only_first_frame