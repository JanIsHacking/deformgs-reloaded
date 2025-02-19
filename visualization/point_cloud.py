from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
import torch
from scene.camera import StaticCamera

@dataclass
class PointCloud:
    points: torch.Tensor
    colors: torch.Tensor

def visualize_point_cloud(point_cloud: PointCloud, title: str = ""):
    assert point_cloud.points.shape[1] == 3

    # Only visualize the first 1000 points
    number_of_points = 100000
    point_cloud.points = point_cloud.points[:number_of_points]
    point_cloud.colors = point_cloud.colors[:number_of_points] if point_cloud.colors is not None else None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(point_cloud.points[:, 0], point_cloud.points[:, 1], point_cloud.points[:, 2], marker='o', s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def visualize_scene(gaussians: PointCloud, cameras: List[StaticCamera]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(gaussians.points[:, 0], gaussians.points[:, 1], gaussians.points[:, 2], c=gaussians.colors, marker='o', s=5)
    camera_centers = torch.stack([camera.camera_center for camera in cameras]).detach().cpu().numpy()
    ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], c='red', marker='x', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()