import random
import torch
import argparse
from submodules.
from submodules.diff_gaussian_rasterization import GaussianRasterizer

class GaussianSplattingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.means = torch.nn.Parameter(torch.randn(1000, 3))
        self.

    def forward(self, camera):
        return self.gaussian_splatting(self.camera_projection(camera))


def train(args):
    print(f"Training 3D reconstruction model for {args.epochs} epochs with Gaussian Splatting...")
    optimizer = torch.optim.Adam(gaussian_model.parameters(), lr=args.learning_rate)
    gaussian_model = GaussianSplattingModel()
    camera_recordings = load_camera_recordings()
    for iteration in range(args.epochs):
        recording = random.choice(camera_recordings)
        camera = recording.camera
        ground_truth_image = recording.get_initial_image()
        image = gaussian_model(camera)
        loss = photometric_loss(image, ground_truth_image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gaussian_model.save(f"model_{iteration}.pth")


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    args = parser.parse_args()
    
    train(args)
