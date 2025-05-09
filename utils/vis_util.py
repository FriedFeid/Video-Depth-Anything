import numpy as np
import os

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from einops import rearrange
import imageio
from natsort import natsorted

from video_depth_anything.video_depth import INFER_LEN

import argparse

#TODO: Include Metric for temporal consistency 

def visualise_data(data_dic, methods, scene_idx, SCENE, FPS, HEIGHT, WIDTH,
                   root='.', Loss_function=None, Loss_name='None', stability_line=0.5, save_gt_stability=False):
    # output_name 
    vis_scene = SCENE[scene_idx]
    vis_name = f'{vis_scene}_{methods}_Vis.mp4'

    # data of RGB and GT
    rgb_vid = data_dic[f'rgb_{SCENE[scene_idx]}']
    gt_depth_vid = data_dic[f'gt_{SCENE[scene_idx]}']

    # Video info 
    frame_count = len(rgb_vid)
    height = rgb_vid.shape[1]
    width = rgb_vid.shape[2]
    stability_x_value = int(width * stability_line)

    # Set up figure
    fig, axs = plt.subplots(nrows=len(methods)+1, ncols=3, figsize=(40, 30))
    
    # Prepare plot 
    Loss_dict = {}
    stability_over_time = {}
    depth_min, depth_max = np.inf, 0.
    error_min, error_max = np.inf, 0.
    Loss_axs = axs[0, 2]
    warm_length = None
    if Loss_function is not None:
        for method in methods:
            # Calculate max
            method_pred = data_dic[method+f'_{SCENE[scene_idx]}']
            method_min, method_max = method_pred.min(), method_pred.max()
            if method_min < depth_min:
                depth_min = method_min
            if method_max > depth_max:
                depth_max = method_max

            try:
                error = np.abs(gt_depth_vid - method_pred)
            except ValueError:
                warm_length = len(gt_depth_vid) - len(method_pred)
                print(f'Assuming wamup length of {warm_length}')
                error = np.abs(gt_depth_vid[warm_length:] - method_pred)
            method_er_min, method_er_max = error.min(), error.max()
            if method_er_min < error_min:
                error_min = method_er_min
            if method_er_max > error_max:
                error_max = method_er_max

            # Calculate Loss
            if 'VDA_s_' in method:
                Loss_dict[method] = Loss_function(method_pred, gt_depth_vid[warm_length:])
            else:
                Loss_dict[method] = Loss_function(method_pred, gt_depth_vid)
            # Set up Lines
            Loss_dict[method+'_line'], = Loss_axs.plot([], [], label=method)
            # Prepare Stablity image
            stability_over_time[method] = np.zeros((height, frame_count), dtype=np.float32)
    else:
        for method in methods:
            # Calculate max
            method_pred = data_dic[method+f'_{SCENE[scene_idx]}']
            method_min, method_max = method_pred.min(), method_pred.max()
            if method_min < depth_min:
                depth_min = method_min
            if method_max > depth_max:
                depth_max = method_max

            try:
                error = np.abs(gt_depth_vid - method_pred)
            except ValueError:
                warm_length = len(gt_depth_vid) - len(method_pred)
                print(f'Assuming wamup length of {warm_length}')
                error = np.abs(gt_depth_vid[warm_length:] - method_pred)
            method_er_min, method_er_max = error.min(), error.max()
            if method_er_min < error_min:
                error_min = method_er_min
            if method_er_max > error_max:
                error_max = method_er_max
            
            # If no Loss is given we calculate MSE
            if 'VDA_s_' in method:
                Loss_dict[method] = np.mean((method_pred - gt_depth_vid[warm_length:])**2, axis=(1,2))
            else:
                Loss_dict[method] = np.mean((method_pred - gt_depth_vid)**2, axis=(1,2))
             # Set up Lines
            Loss_dict[method+'_line'], = Loss_axs.plot([], [], label=method)
            # Prepare Stablity image
            stability_over_time[method] = np.zeros((height, frame_count), dtype=np.float32)
    
    # Prepare GT Stability image:
        if save_gt_stability:
            gt_stability = np.zeros((height, frame_count), dtype=np.float32)
    
    Loss_min, Loss_max = np.inf , 0.
    for key in methods:
        min_, max_ = Loss_dict[key].min(), Loss_dict[key].max()
        if min_ < Loss_min:
            Loss_min = min_
        if max_ > Loss_max:
            Loss_max = max_ 

    Loss_axs.set_xlim(0, frame_count)
    Loss_axs.set_ylim( Loss_min, Loss_max)

    # Setup video writer variables
    writer = imageio.get_writer(os.path.join(root, vis_name), fps=FPS, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    for t in tqdm(range(len(rgb_vid))):
        # Start with GT viedo
        axs[0, 0].clear()
        axs[0, 0].imshow(rgb_vid[t])
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        # GT depth
        axs[0, 1].clear()
        axs[0, 1].imshow(gt_depth_vid[t], cmap='Spectral', vmin=depth_min, vmax=depth_max)
        axs[0, 1].set_title('GT Scene')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])

        # GT stability plot
        if save_gt_stability:
            gt_stability[:, t] = gt_depth_vid[t, :, stability_x_value]

        #Loss plot
        axs[0, 2].legend()
        if Loss_function is None:
            axs[0, 2].set_title(f'MSE_Loss')
        else:
            axs[0, 2].set_title(f'{Loss_name}')

        for i, method in enumerate(methods):
            if 'VDA_s_' in method and t <= warm_length:
                # Plot Frames
                method_pred = np.zeros((warm_length, HEIGHT, WIDTH))
                error_map = np.zeros((HEIGHT, WIDTH))
                # Stability over time
                satbility = stability_over_time[method]
                t_warmup = t
            else:
                if 'VDA_s_' in method:
                    Loss_dict[method+'_line'].set_data(np.arange(t-t_warmup) + t_warmup, Loss_dict[method][:t-t_warmup])

                    # Plot Frames
                    method_pred = data_dic[method+f'_{SCENE[scene_idx]}']
                    error_map = np.abs(gt_depth_vid[t] - method_pred[t-t_warmup])
                    # Stability over time
                    satbility = stability_over_time[method]
                    satbility[:, t] = method_pred[t-t_warmup, :, stability_x_value]
                    stability_over_time[method] = satbility
                else:
                    Loss_dict[method+'_line'].set_data(np.arange(t), Loss_dict[method][:t])

                    # Plot Frames
                    method_pred = data_dic[method+f'_{SCENE[scene_idx]}']
                    error_map = np.abs(gt_depth_vid[t] - method_pred[t])
                    # Stability over time
                    satbility = stability_over_time[method]
                    satbility[:, t] = method_pred[t, :, stability_x_value]
                    stability_over_time[method] = satbility
            # Abs Err. to gt
            axs[i+1, 0].clear()
            axs[i+1, 0].imshow(error_map, cmap='RdYlGn', vmin=error_min, vmax=error_max)
            axs[i+1, 0].set_xticks([])
            axs[i+1, 0].set_yticks([])
            # Prediction
            axs[i+1, 1].clear()
            if 'VDA_s_' in method:
                axs[i+1, 1].imshow(method_pred[t-t_warmup], cmap='Spectral', vmin=depth_min, vmax=depth_max)
            else:
                axs[i+1, 1].imshow(method_pred[t], cmap='Spectral', vmin=depth_min, vmax=depth_max)
            axs[i+1, 1].set_xticks([])
            axs[i+1, 1].set_yticks([])
            axs[i+1, 1].set_title(method)
            # Stability over time
            axs[i+1, 2].clear()
            axs[i+1, 2].imshow(satbility, cmap='Spectral', vmin=depth_min, vmax=depth_max)
            axs[i+1, 2].set_xticks([])
            axs[i+1, 2].set_yticks([])

        fig.canvas.draw()
        matplotlib_frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        writer.append_data(matplotlib_frame)
    
    writer.close()

    
    # Save gt Stability Line
    plt.close('all')
    if save_gt_stability:
        gt_name = f'{vis_scene}_GT_tmpc_Vis.png'
        plt.imshow(gt_stability, cmap='Spectral', vmin=depth_min, vmax=depth_max)
        plt.savefig(os.path.join(root, gt_name))


## Harman Results plot 

def visualise_money_plot(data_dic, root, methods, SCENE, FPS, HEIGHT, WIDTH):
        # output_name 
    vis_scene = SCENE[0]
    vis_name = f'{vis_scene}_{methods}_Money.mp4'

    # data of RGB and GT
    rgb_vid = data_dic[f'rgb_{SCENE[0]}']
    gt_depth_vid = data_dic[f'gt_{SCENE[0]}']

    # Video info 
    frame_count = len(rgb_vid)
    height = rgb_vid.shape[1]
    width = rgb_vid.shape[2]


    # Set up figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    
    # Prepare plot 
    Loss_dict = {}
    stability_over_time = {}
    depth_min, depth_max = np.inf, 0.
    error_min, error_max = np.inf, 0.
    warm_length = None

    # Calculate max
    for method in methods:
        method_pred = data_dic[method+f'_{SCENE[0]}']
        method_min, method_max = method_pred.min(), method_pred.max()
        if method_min < depth_min:
            depth_min = method_min
        if method_max > depth_max:
            depth_max = method_max

        try:
                error = np.abs(gt_depth_vid - method_pred)
        except ValueError:
            warm_length = len(gt_depth_vid) - len(method_pred)
            print(f'Assuming wamup length of {warm_length}')
            error = np.abs(gt_depth_vid[warm_length:] - method_pred)

    # Setup video writer variables
    writer = imageio.get_writer(os.path.join(root, vis_name), fps=FPS, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    for t in tqdm(range(len(rgb_vid))):
        # Start with GT viedo
        axs[0].clear()
        axs[0].imshow(rgb_vid[t])
        axs[0].set_title('Original RGB Video')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        # GT depth
        axs[1].clear()
        axs[1].imshow(gt_depth_vid[t], cmap='Spectral', vmin=depth_min, vmax=depth_max)
        axs[1].set_title('GT Scene')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        #Loss plo
        for i, method in enumerate(methods):
            if 'VDA_s_' in method and t <= warm_length:
                # Plot Frames
                method_pred = np.zeros((warm_length, HEIGHT, WIDTH))
                error_map = np.zeros((HEIGHT, WIDTH))
                # Stability over time
                t_warmup = t
            else:
                if 'VDA_s_' in method:

                    # Plot Frames
                    method_pred = data_dic[method+f'_{SCENE[0]}']
                    error_map = np.abs(gt_depth_vid[t] - method_pred[t-t_warmup])


            # Prediction
            if 'VDA_s_' in method:
                axs[2].clear()

                axs[2].imshow(method_pred[t-t_warmup], cmap='Spectral', vmin=depth_min, vmax=depth_max)

                axs[2].set_xticks([])
                axs[2].set_yticks([])
                axs[2].set_title('Ours')
            else:
                break


        fig.canvas.draw()
        matplotlib_frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        writer.append_data(matplotlib_frame)
    
    writer.close()


