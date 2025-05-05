import torch
import numpy as np 
import matplotlib.pyplot as plt
from einops import rearrange
import open3d as o3d
from PIL import Image

def torch_rgb_to_numpy_rgb(torch_tensor):
    return rearrange(torch_tensor, 'c h w -> h w c').numpy()

def generate_gif(torch_tensor_imgaes, path):
    # Rescale to [0, 255] and convert to uint8
    torch_tensor_imgaes = (torch_tensor_imgaes * 255).clamp(0, 255).byte()
    num_frames = torch_tensor_imgaes.size()[0]

    # Permute to [frame, height, width, channels]
    frames = rearrange(torch_tensor_imgaes, 'f c h w -> f h w c').numpy()
    # Erstelle eine Liste von PIL-Images
    images = [Image.fromarray(frame) for frame in frames]

    # Speichere als GIF
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(num_frames / 25.),  # Dauer pro Frame in ms (~25 FPS)
        loop=0
    )

def plot_scene_image(sample, image_idx):
    images_in_scene = len(sample['image'])
    print('Images in scene ', images_in_scene)
    print('')
    for type in ['image', 'depth']:
        print(type)
        print('Resolution: ', sample[type].size())
        print('Min Value: ', sample[type].min())
        print('Max Value: ', sample[type].max())
        print('dtype: ', sample[type].dtype)
        print('')
    
    print('Camera Intrinsics:')
    print(sample['intrinsics'][image_idx])
    print(sample['intrinsics'].shape)
    print('')

    print('Camera Extrinsics')
    print(sample['extrinsics'][image_idx])
    print(sample['extrinsics'].shape)
    print('')

    rgb_image = torch_rgb_to_numpy_rgb(sample['image'][image_idx])
    h, w, c = rgb_image.shape
    depth_image = sample['depth'][image_idx].numpy()
    depth_valid = sample['valid_depth'][image_idx].numpy()

    fig, axs = plt.subplots(1, 3, figsize=(8,4))

    axs[0].imshow(rgb_image, extent=[0, w, h, 0])
    axs[0].set_title(f'RGB image {image_idx}')
    axs[0].axis('off')

    axs[1].imshow(depth_valid, cmap='Greens', vmin=0., vmax=1.,)
    axs[1].set_title(f'Valid depth Values')
    axs[1].axis('off')

    im = axs[2].imshow(depth_image, cmap='Spectral', vmin=sample['depth'].min(), 
                       vmax=sample['depth'].max(), extent=[0, w, h, 0])
    axs[2].set_title('Depth')
    axs[2].axis('off')

    # Hinzufügen der Matplotlib colorbar axs
    cbar_ax = fig.add_axes([0.99, 0.23, 0.02, 0.57])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)

    fig.tight_layout()
    plt.show()

def project_image_to_pointcloud(sample, image_idx, max_depth=1_000, 
                                Cam_to_World=False, Flip_to_open3d=np.diag([1., 1., 1., 1.])):
    # Load needed Data
    rgb_image = (torch_rgb_to_numpy_rgb(sample['image'][image_idx]) * 255.).astype(np.uint8)
    depth_map = sample['depth'][image_idx].numpy()
    valid_depth = sample['valid_depth'][image_idx].numpy()
    Intrinsics = sample['intrinsics'][image_idx].numpy()
    Extrinsics = sample['extrinsics'][image_idx].numpy() # Da hier Cam to World

    # Masking of invalid by setting to maximal Depth 
    # depth = np.where(valid_depth, depth_map, 0.)
    depth = depth_map

    assert rgb_image.shape[:2] == depth_map.shape == valid_depth.shape

    # Create Camera in Open3D
    h, w = depth_map.shape
    fx, fy = Intrinsics[0, 0], Intrinsics[1, 1]
    cx, cy = Intrinsics[0, 2], Intrinsics[1, 2]
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    # Transform Images into o3d Format
    o3d_color = o3d.geometry.Image(np.ascontiguousarray(rgb_image))
    o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth,
                                                              depth_scale=1.0,
                                                              convert_rgb_to_intensity=False,
                                                              depth_trunc=max_depth-1e-4)
    # To actually clip max value depths 
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)
    # Invert if Extrinsics are World_to_Cam to get Cam_to_World
    if not Cam_to_World:
        Extrinsics = np.linalg.inv(Extrinsics)
    pcd.transform(Extrinsics)
    pcd.transform(Flip_to_open3d)

    return pcd

def project_image_to_3d(sample, image_idx, save_path, max_depth=1_000.,
                        Cam_to_World=False, Flip_to_open3d=np.diag([1., 1., 1., 1.])):
    pcd = project_image_to_pointcloud(sample, image_idx, max_depth, 
                                      Cam_to_World, Flip_to_open3d)
    o3d.io.write_point_cloud(save_path, pcd)

def project_scene_to_3d(sample, multiple_img_idx, save_path, max_depth=1_000.,
                        Cam_to_World=False, Flip_to_open3d=np.diag([1., 1., 1., 1.]),
                        Visualise_cam_pos=False):
    # cam_frames = []
    # for idx in multiple_img_idx:
    #     T = np.linalg.inv(sample['extrinsics'][idx].numpy())
    #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    #     frame.transform(T)
    #     cam_frames.append(frame)

    # # Merge all camera frames to one mesh
    # merged_frames = cam_frames[0]
    # for frame in cam_frames[1:]:
    #     merged_frames += frame

    # o3d.io.write_triangle_mesh(save_path.replace('.ply', '_camframes.ply'), merged_frames)
    if Visualise_cam_pos:
        positions = []

        for idx in multiple_img_idx:
            T = sample['extrinsics'][idx].numpy()
            if not Cam_to_World:
                T = np.linalg.inv(T)

            T = T @ Flip_to_open3d  # bringe Pose in Open3D-Konvention

            cam_position = T[:3, 3]  # extrahiere Kameraursprung
            positions.append(cam_position)

        positions = np.stack(positions)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 2], positions[:, 1], marker='o')
        
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        # zentrierte Skalierung
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_zlim(mid_y - max_range, mid_y + max_range)
        ax.set_ylim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel("X (right)")
        ax.set_ylabel("Z (forward")   # Open3D-style
        ax.set_zlabel("Y (down)")
        plt.show()

    merged_pcd = o3d.geometry.PointCloud()
    for idx in multiple_img_idx:
        merged_pcd += project_image_to_pointcloud(sample, idx, max_depth,
                                                  Cam_to_World, Flip_to_open3d)
    o3d.io.write_point_cloud(save_path, merged_pcd)
