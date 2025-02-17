import numpy as np
from scene.camera import load_test_static_camera_recordings, load_training_static_camera_recordings
from matplotlib import pyplot as plt
from model.gaussian_splatting import GaussianSplattingModel
import torch
from PIL import Image


def main():
    training_camera_recordings = load_training_static_camera_recordings()
    test_camera_recordings = load_test_static_camera_recordings()
    gaussian_model = GaussianSplattingModel(max_sh_degree=3, bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
    gaussian_model.load("output/checkpoints/model.pth")
    camera = training_camera_recordings[10].camera
    image, radii = gaussian_model([camera])
    test_image = np.array(Image.open(training_camera_recordings[10].get_initial_image().file_path).convert("RGB"))
    plt.imshow(image[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()


if __name__ == "__main__":
    main()