import matplotlib.pyplot as plt
import torch

def visualize_point_cloud(point_cloud: torch.Tensor):
    assert point_cloud.shape[1] == 3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], marker='o', s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
