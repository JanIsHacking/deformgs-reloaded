import math
from typing import Dict, List
import os
import json

import numpy as np
import torch
import torch.nn as nn


from PIL import Image
import torchvision.transforms as T


def focal_length_to_field_of_view(focal_length, pixels):
    return 2*math.atan(pixels/(2*focal_length))


def getProjectionMatrixPanopto(f_x,f_y,c_x,c_y,width,height,znear,zfar):
    opengl_proj = torch.tensor([[2 * f_x / width, 0.0, -(width - 2 * c_x) / width, 0.0],
                                [0.0, 2 * f_y / height, -(height - 2 * c_y) / height, 0.0],
                                [0.0, 0.0, zfar / (zfar - znear), -(zfar * znear) / (zfar - znear)],
                                [0.0, 0.0, 1.0, 0.0]]).float().unsqueeze(0).transpose(1, 2).cuda()
    return opengl_proj


class StaticCamera(nn.Module):
    def __init__(self, camera_id: int, rotation: torch.Tensor, translation: torch.Tensor, focal_length_x: float, focal_length_y: float, image_width: int, image_height: int, cx: float, cy: float, angle_x: float, angle_y: float):
        self.camera_id = camera_id
        self.rotation = rotation
        self.translation = translation
        self.focal_length_x = focal_length_x
        self.focal_length_y = focal_length_y
        self.image_width = image_width
        self.image_height = image_height
        self.cx = cx
        self.cy = cy
        self.angle_x = angle_x
        self.angle_y = angle_y

        self.fov_x = focal_length_to_field_of_view(focal_length_x, image_width)
        self.fov_y = focal_length_to_field_of_view(focal_length_y, image_height)

        # TODO: Verify if this transformation is correct
        Rt = np.eye(4)
        Rt[:3, :3] = rotation.T.numpy()
        Rt[:3, 3] = -Rt[:3, :3] @ translation.numpy()
        Rt[1] *= -1
        Rt[2] *= -1
        self.world_view_transform = torch.tensor(np.float32(Rt)).transpose(0,1).cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3].cuda()
        
        # TODO: Add as hyperparameter
        znear = 1
        zfar = 100
        # TODO: Verify if this transformation is correct
        self.projection_matrix = getProjectionMatrixPanopto(f_x=focal_length_x,f_y=focal_length_y,c_x=cx,c_y=cy,width=image_width,height=image_height,znear=znear,zfar=zfar)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix)).squeeze(0).cuda()

class ImageFrame:
    def __init__(self, file_path: str, time: float):
        self.file_path = file_path
        self.time = time

    def load(self):
        # Discarding the alpha channel
        image = np.array(Image.open(self.file_path).convert("RGB"))
        image_tensor = (torch.from_numpy(image).float() / 255.0).permute(2, 0, 1)
        return image_tensor


class StaticCameraRecording:
    def __init__(self, camera: StaticCamera, images: List[ImageFrame]):
        self.camera = camera
        self.images = images

    def __repr__(self) -> str:
        return f"CameraRecording(camera_id={self.camera.camera_id}, images={len(self.images)})"
    
    def get_initial_image(self):
        return self.images[0]
    
def _load_static_camera_recordings(data_dir: str, transforms_file: str) -> List[StaticCameraRecording]:
    with open(os.path.join(data_dir, transforms_file), "r") as f:
        transforms = json.load(f)
    recordings: Dict[int, List[ImageFrame]] = {}
    
    for frame in transforms["frames"]:
        file_path = os.path.join(data_dir, frame["file_path"])
        time = frame["time"]
        parts = file_path.split("/")[-1].split("_")
        camera_id = int(parts[1])
        transformation_matrix = torch.tensor(frame["transform_matrix"])
        rotation = transformation_matrix[:3, :3]
        translation = transformation_matrix[:3, 3]
        
        if camera_id not in recordings:
            recordings[camera_id] = []
        
        recordings[camera_id].append(ImageFrame(file_path, time))
    
    camera_recordings: List[StaticCameraRecording] = []

    for camera_id, images in recordings.items():
        camera = StaticCamera(
            camera_id=camera_id,
            rotation=rotation,
            translation=translation,
            focal_length_x=transforms["fl_x"],
            focal_length_y=transforms["fl_y"],
            image_width=transforms["w"],
            image_height=transforms["h"],
            cx=transforms["cx"],
            cy=transforms["cy"],
            angle_x=transforms["camera_angle_x"],
            angle_y=transforms["camera_angle_y"]
        )
        camera_recordings.append(StaticCameraRecording(camera, images))
    
    return camera_recordings

    
def load_test_static_camera_recordings() -> List[StaticCameraRecording]:
    test_data_dir = os.path.join("data", "synthetic", "scene_1")
    return _load_static_camera_recordings(test_data_dir, "transforms_test.json")

def load_training_static_camera_recordings() -> List[StaticCameraRecording]:
    train_data_dir = os.path.join("data", "synthetic", "scene_1")
    return _load_static_camera_recordings(train_data_dir, "transforms_train.json")

