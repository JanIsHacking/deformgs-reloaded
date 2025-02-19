import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
import numpy as np
from train import load_initial_point_cloud
from visualization.point_cloud import visualize_scene
from model.gaussian_splatting import GaussianSplattingModel

from scene.camera import load_training_cameras


def main():
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    initial_point_cloud = load_initial_point_cloud(os.path.join('data', 'synthetic', 'scene_1', 'initial_point_cloud.ply'))
    gaussian_model = GaussianSplattingModel(
        max_sh_degree=3, 
        bg_color=bg_color, 
        initial_point_cloud=initial_point_cloud,
        initial_num_gaussians=15000
    )
    training_cameras = load_training_cameras()
    for camera in training_cameras:
        #visualize_scene(initial_point_cloud, [camera])
        images, radii = gaussian_model([camera])
        plt.imshow(images[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.show()
        plt.imshow(camera.get_initial_recording().get_initial_image().load().permute(1, 2, 0).numpy())
        plt.show()


if __name__ == "__main__":
    main()