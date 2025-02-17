from dataclasses import dataclass
from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from scene.camera import StaticCamera, StaticCameraRecording, load_training_static_camera_recordings
from model.gaussian_splatting import GaussianSplattingModel

def collate_fn(batch):
    cameras, images = zip(*batch)  # Unpack the batch
    return list(cameras), torch.stack(images)


class StaticCameraRecordingDataset(Dataset):
    def __init__(self, camera_recordings: List[StaticCameraRecording]):
        self.camera_recordings = camera_recordings

    def __len__(self):
        return len(self.camera_recordings)

    def __getitem__(self, idx) -> Tuple[StaticCamera, torch.Tensor]:
        recording: StaticCameraRecording = self.camera_recordings[idx]
        camera = recording.camera
        ground_truth_image = recording.get_initial_image()
        image_tensor = ground_truth_image.load().cuda()
        return camera, image_tensor


def photometric_loss(images: torch.Tensor, ground_truth_images: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(images, ground_truth_images)


def train(args):
    print(f"Training 3D reconstruction model for {args.epochs} epochs with Gaussian Splatting...")
    # TODO: Check what the background is needed for and if it should be a parameter
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    gaussian_model = GaussianSplattingModel(max_sh_degree=args.max_sh_degree, bg_color=bg_color)
    
    training_camera_recordings = load_training_static_camera_recordings()
    training_dataset = StaticCameraRecordingDataset(training_camera_recordings)
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    progress_bar = tqdm(range(args.epochs), desc="Training")
    for epoch in progress_bar:
        total_loss = 0.0
        num_batches = 0
        for batch_idx, (cameras, ground_truth_images) in enumerate(training_dataloader):
            images, radii = gaussian_model(cameras)
            loss = photometric_loss(images, ground_truth_images)
            
            gaussian_model.optimizer.zero_grad()
            loss.backward()
            gaussian_model.optimizer.step()

            if (epoch*len(training_dataloader) + batch_idx) % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss {loss.item():.4f}")

            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        progress_bar.set_postfix(loss=avg_loss)
    
    gaussian_model.save("output/checkpoints/model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_sh_degree", type=int, default=3, help="Maximum SH degree")
    args = parser.parse_args()
    
    train(args)
