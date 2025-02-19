import os
import numpy as np
from scene.camera import load_test_cameras, load_training_cameras
from matplotlib import pyplot as plt
from model.gaussian_splatting import GaussianSplattingModel, PointCloud
import torch
from PIL import Image
import glob
from datetime import datetime, timedelta

from train import load_initial_point_cloud
from visualization.point_cloud import visualize_point_cloud


def main():
    training_cameras = load_training_cameras()
    test_cameras = load_test_cameras()
    initial_point_cloud = load_initial_point_cloud(os.path.join('data', 'synthetic', 'scene_1', 'initial_point_cloud.ply'))
    visualize_point_cloud(initial_point_cloud, "Initial Point Cloud")
    gaussian_model = GaussianSplattingModel(max_sh_degree=3, bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
    
    checkpoint_files = glob.glob("output/checkpoints/model_*.pth")
    newest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    timestamp = int(newest_checkpoint.split('_')[-1].split('.')[0])
    readable_time = (datetime.fromtimestamp(timestamp)- timedelta(hours=5)).strftime('%d-%m-%Y %H:%M:%S')
    print(f"Loading model checkpoint from {readable_time}")
    gaussian_model.load(newest_checkpoint)
    visualize_point_cloud(PointCloud(gaussian_model.means.detach().cpu().numpy(), None), "Loaded Point Cloud")
    
    camera = training_cameras[10]
    image, radii = gaussian_model([camera])
    test_image = np.array(Image.open(camera.get_initial_recording().get_initial_image().file_path).convert("RGB"))
    plt.imshow(image[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()

    plt.imshow(test_image)
    plt.show()


if __name__ == "__main__":
    main()