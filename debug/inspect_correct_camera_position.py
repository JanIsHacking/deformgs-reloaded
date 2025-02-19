import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
import numpy as np
from train import load_initial_point_cloud
from visualization.point_cloud import visualize_scene

from scene.camera import load_training_cameras


def main():
    training_cameras = load_training_cameras()
    initial_point_cloud = load_initial_point_cloud(os.path.join('data', 'synthetic', 'scene_1', 'initial_point_cloud.ply'))
    for camera in training_cameras:
        visualize_scene(initial_point_cloud, [camera])
        plt.imshow(camera.get_initial_recording().get_initial_image().load().permute(1, 2, 0).numpy())
        plt.show()


if __name__ == "__main__":
    main()