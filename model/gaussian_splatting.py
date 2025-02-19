from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import List, Optional

from scene.camera import StaticCamera
from utils.sh_utils import RGB2SH, eval_sh
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from scipy.spatial.distance import cdist

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

@dataclass
class PointCloud:
    points: torch.Tensor
    colors: torch.Tensor


class GaussianSplattingModel(torch.nn.Module):
    def __init__(self, 
                 max_sh_degree: int, 
                 bg_color: torch.Tensor, 
                 initial_point_cloud: Optional[PointCloud] = None, 
                 initial_num_gaussians: int = 1000
        ):
        super().__init__()
        
        self.active_sh_degree = 0
        self.max_sh_degree = max_sh_degree  
        self.bg_color = bg_color

        # Creating means 2D as variable not as parameter because we are only interested in its gradient
        # not the means themselves
        if initial_point_cloud is not None:
            self.means_2d = torch.zeros((np.min([initial_num_gaussians, initial_point_cloud.points.shape[0]]), 3), dtype=torch.float32, requires_grad=True, device="cuda")
        else:
            self.means_2d = torch.zeros((initial_num_gaussians, 3), dtype=torch.float32, requires_grad=True, device="cuda")
        self.means_2d.retain_grad()

        if initial_point_cloud is not None:
            self._initialize_from_point_cloud(initial_point_cloud, initial_num_gaussians)
        else:
            self._initialize_randomly(initial_num_gaussians)

        feature_learning_rate = 0.0025
        parameters = [
            { 'params': self.means, 'lr': 0.000016, 'name': 'means' },
            { 'params': self.scales, 'lr': 0.005, 'name': 'scales' },
            { 'params': self.rotations, 'lr': 0.001, 'name': 'rotations' },
            { 'params': self.opacities, 'lr': 0.05, 'name': 'opacities' },
            { 'params': self.sh_direct_current, 'lr': feature_learning_rate, 'name': 'sh_direct_current' },
            { 'params': self.sh_high_order, 'lr': feature_learning_rate / 20.0, 'name': 'sh_high_order' }
        ]
        self.optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)

        self.means_lr_function = get_expon_lr_func(0.000016, 0.00000016, lr_delay_mult=0.1, max_steps=60000)
    
    def _initial_scales_from_point_cloud(self, point_cloud: PointCloud):
        pairwise_distances = cdist(point_cloud.points, point_cloud.points, metric='euclidean')
        K = 10
        knn_distances = np.sort(pairwise_distances, axis=1)[:, 1:K+1]  # Excluding self distance (diagonal)
        mean_knn_distances = np.mean(knn_distances, axis=1)

        dist = torch.tensor(mean_knn_distances, dtype=torch.float32, device="cuda")
        dist = torch.clamp_min(dist, 0.0000001)  # Ensure no zeros
        #scales = torch.log(torch.sqrt(dist))[..., None].repeat(1, 3)
        scales = torch.sqrt(dist)[..., None].repeat(1, 3)
        return scales
    
    def _initialize_from_point_cloud(self, point_cloud: PointCloud, initial_num_gaussians: int):
        # Randomly sample points from the point cloud
        if initial_num_gaussians < point_cloud.points.shape[0]:
            num_points = point_cloud.points.shape[0]
            indices = torch.randperm(num_points)[:initial_num_gaussians]
            sampled_points = point_cloud.points[indices]
            point_cloud.points = sampled_points
            point_cloud.colors = point_cloud.colors[indices]
        self.means = torch.nn.Parameter(torch.tensor(point_cloud.points, dtype=torch.float32, device="cuda"))
        
        # Add some noise to the scales and ensure they're not too small
        scales = self._initial_scales_from_point_cloud(point_cloud)
        #scales = scales + torch.randn_like(scales) * 0.1
        #scales = torch.clamp(scales, min=-3.0, max=1.0)
        self.scales = torch.nn.Parameter(scales)
        
        # Initialize rotations with small random perturbations
        initial_rotations = torch.randn((point_cloud.points.shape[0], 4), device="cuda") * 0.1
        initial_rotations[:, 0] = 1.0  # Set w component to 1
        initial_rotations = initial_rotations / initial_rotations.norm(dim=-1, keepdim=True)  # Normalize
        self.rotations = torch.nn.Parameter(initial_rotations)
        
        # Initialize opacities in logit space
        initial_opacities = torch.full((point_cloud.points.shape[0], 1), 0.1, device="cuda")
        self.opacities = torch.nn.Parameter(initial_opacities)

        # Rest of the code remains the same
        colors = torch.tensor(point_cloud.colors, dtype=torch.float32, device="cuda")
        sh_direct_current = RGB2SH(colors)
        sh_higher_order = torch.zeros((colors.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3), device="cuda")
        self.sh_direct_current = torch.nn.Parameter(sh_direct_current.unsqueeze(1))
        self.sh_high_order = torch.nn.Parameter(sh_higher_order)

    def _initialize_randomly(self, initial_num_gaussians: int):
        self.means = torch.nn.Parameter(torch.randn(initial_num_gaussians, 3, device="cuda"))
        self.scales = torch.nn.Parameter(torch.randn(initial_num_gaussians, 1, device="cuda"))
        self.rotations = torch.nn.Parameter(torch.randn(initial_num_gaussians, 4, device="cuda"))
        self.opacities = torch.nn.Parameter(torch.randn(initial_num_gaussians, 1, device="cuda"))

        colors = torch.zeros(initial_num_gaussians, 3, device="cuda")
        sh_direct_current = RGB2SH(colors)
        sh_higher_order = torch.zeros((colors.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3), device="cuda")
        self.sh_direct_current = torch.nn.Parameter(sh_direct_current.unsqueeze(1))
        self.sh_high_order = torch.nn.Parameter(sh_higher_order)

    def forward(self, cameras: List[StaticCamera], scaling_modifier = 1.0) -> torch.Tensor:
        batch_images = []
        batch_radii = []
        for camera in cameras:
            tanfovx = camera.image_width/(2*camera.focal_length_x)
            tanfovy = camera.image_height/(2*camera.focal_length_y)
            rasterizer = GaussianRasterizer(GaussianRasterizationSettings(
                image_height=int(camera.image_height),
                image_width=int(camera.image_width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=self.bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=camera.world_view_transform.cuda(),
                projmatrix=camera.full_proj_transform.cuda(),
                sh_degree=self.active_sh_degree,
                campos=camera.camera_center.cuda(),
                prefiltered=False,
                debug=False
            ))
            image, radii = rasterizer(
                self.means,
                self.means_2d,
                self.opacities,
                None,
                self._precompute_colors(camera),
                self.scales,
                self.rotations,
                None
            )
            batch_images.append(image)
            batch_radii.append(radii)
        return torch.stack(batch_images), torch.stack(batch_radii)
    
    def update_learning_rate(self, iteration: int):
        for param in self.optimizer.param_groups:
            if param['name'] == 'means':
                param['lr'] = self.means_lr_function(iteration)
    
    def _get_shs(self):
        return torch.cat([self.sh_direct_current, self.sh_high_order], dim=1)
    
    def _precompute_colors(self, camera: StaticCamera) -> torch.Tensor:
        """
        Precompute the colors for the given camera.

        Args:
            camera: StaticCamera

        Returns:
            torch.Tensor: Precomputed colors [..., 3]
        """
        viewing_directions = self.means - camera.camera_center
        normalized_viewing_directions = viewing_directions / torch.norm(viewing_directions, dim=1, keepdim=True)
        precomputed_colors = eval_sh(self.active_sh_degree, self._get_shs().permute(0,2,1), normalized_viewing_directions)
        precomputed_colors += 0.5
        precomputed_colors = torch.clamp_min(precomputed_colors, 0.0)
        return precomputed_colors
    
    def _increment_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def save(self, path: str):
        torch.save({
            "means": self.means.detach().cpu(),
            "scales": self.scales.detach().cpu(),
            "rotations": self.rotations.detach().cpu(),
            "opacities": self.opacities.detach().cpu(),
            "sh_direct_current": self.sh_direct_current.detach().cpu(),
            "sh_high_order": self.sh_high_order.detach().cpu(),
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.means.data = checkpoint["means"].cuda()
        self.scales.data = checkpoint["scales"].cuda()
        self.rotations.data = checkpoint["rotations"].cuda()
        self.opacities.data = checkpoint["opacities"].cuda()
        self.sh_direct_current.data = checkpoint["sh_direct_current"].cuda()
        self.sh_high_order.data = checkpoint["sh_high_order"].cuda()