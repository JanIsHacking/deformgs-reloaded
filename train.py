import os
import time
from typing import List, Tuple
import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from plyfile import PlyData
import argparse
from tqdm import tqdm
from scene.camera import StaticCamera, load_training_cameras
from model.gaussian_splatting import GaussianSplattingModel, PointCloud
from visualization.point_cloud import visualize_point_cloud

def collate_fn(batch):
    cameras, images = zip(*batch)  # Unpack the batch
    return list(cameras), torch.stack(images)


class StaticCameraDataset(Dataset):
    def __init__(self, cameras: List[StaticCamera]):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx) -> Tuple[StaticCamera, torch.Tensor]:
        camera: StaticCamera = self.cameras[idx]
        ground_truth_image = camera.get_initial_recording().get_initial_image()
        image_tensor = ground_truth_image.load().cuda()
        return camera, image_tensor


def photometric_loss(images: torch.Tensor, ground_truth_images: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(images, ground_truth_images)

def load_initial_point_cloud(path: str):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return PointCloud(positions, colors)

def train(args):
    print(f"Training 3D reconstruction model for {args.epochs} epochs with Gaussian Splatting...")
    # TODO: Check what the background is needed for and if it should be a parameter
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    initial_point_cloud = load_initial_point_cloud(os.path.join('data', 'synthetic', 'scene_1', 'initial_point_cloud.ply'))
    gaussian_model = GaussianSplattingModel(
        max_sh_degree=args.max_sh_degree, 
        bg_color=bg_color, 
        initial_point_cloud=initial_point_cloud,
        initial_num_gaussians=args.initial_num_gaussians
    )
    visualize_point_cloud(PointCloud(gaussian_model.means.detach().cpu().numpy(), None))
    
    training_cameras = load_training_cameras()
    training_dataset = StaticCameraDataset(training_cameras)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    progress_bar = tqdm(range(args.epochs), desc="Training")
    for epoch in progress_bar:
        #gaussian_model.update_learning_rate(epoch)

        total_loss = 0.0
        for batch_idx, (cameras, ground_truth_images) in enumerate(training_dataloader):
            images, radii = gaussian_model(cameras)
            loss = torch.abs((images - ground_truth_images)).mean()
            
            loss.backward()
            with torch.no_grad():
                if (epoch*len(training_dataloader) + batch_idx) % 500 == 0:
                    # Every 100 steps we densify and prune
                    wandb.log({"loss": loss.item()})
                    wandb.log({"means_x_gradient": gaussian_model.means.grad[:, 0].norm()})
                    wandb.log({"means_y_gradient": gaussian_model.means.grad[:, 1].norm()})
                    wandb.log({"means_z_gradient": gaussian_model.means.grad[:, 2].norm()})
                    wandb.log({"means_gradient_norm": gaussian_model.means.grad.norm()})
                    wandb.log({"scales_gradient_norm": gaussian_model.scales.grad.norm()})
                    wandb.log({"rotations_gradient_norm": gaussian_model.rotations.grad.norm()})
                    wandb.log({"opacities_gradient_norm": gaussian_model.opacities.grad.norm()})
                    wandb.log({"sh_direct_current_gradient_norm": gaussian_model.sh_direct_current.grad.norm()})
                    wandb.log({"sh_high_order_gradient_norm": gaussian_model.sh_high_order.grad.norm()})
            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad()
            if (epoch*len(training_dataloader) + batch_idx) % 100 == 0:
                # Every 100 steps we densify and prune
                pass

            total_loss += loss.item()
        avg_loss = total_loss / batch_idx
        progress_bar.set_postfix(loss=avg_loss)
    
    gaussian_model.save(f"output/checkpoints/model_{int(time.time())}.pth")
    wandb.finish()


def log_hyper_parameters(args: argparse.Namespace):
    for arg in vars(args):
        wandb.config[arg] = getattr(args, arg)


if __name__ == "__main__":
    wandb.init(project="DeformGS Reloaded",name="Scene 1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_sh_degree", type=int, default=3, help="Maximum SH degree")
    parser.add_argument("--initial_num_gaussians", type=int, default=15000, help="Number of initial gaussians")
    args = parser.parse_args()
    log_hyper_parameters(args)
    
    train(args)
