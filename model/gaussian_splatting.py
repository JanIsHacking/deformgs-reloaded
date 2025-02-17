import torch
from typing import List

from scene.camera import StaticCamera
from utils.sh_utils import RGB2SH, eval_sh
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings


class GaussianSplattingModel(torch.nn.Module):
    def __init__(self, max_sh_degree: int, bg_color: torch.Tensor):
        super().__init__()
        
        self.active_sh_degree = 0
        self.max_sh_degree = max_sh_degree  
        self.bg_color = bg_color

        self.means = torch.nn.Parameter(torch.randn(1000, 3, device="cuda"))
        self.scales = torch.nn.Parameter(torch.randn(1000, 1, device="cuda"))
        self.rotations = torch.nn.Parameter(torch.randn(1000, 4, device="cuda"))
        self.opacities = torch.nn.Parameter(torch.randn(1000, 1, device="cuda"))
        # TODO: Separate SH parameters into direct current and rest
        colors = torch.zeros(1000, 3, device="cuda")
        sh_direct_current = RGB2SH(colors)
        sh_higher_order = torch.zeros((colors.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3)).float()
        self.sh_direct_current = torch.nn.Parameter(sh_direct_current.unsqueeze(1).cuda())
        self.sh_high_order = torch.nn.Parameter(sh_higher_order.cuda())

        feature_learning_rate = 0.0025
        parameters = [
            { 'params': self.means, 'lr': 0.000016 },
            { 'params': self.scales, 'lr': 0.005 },
            { 'params': self.rotations, 'lr': 0.001 },
            { 'params': self.opacities, 'lr': 0.05 },
            { 'params': self.sh_direct_current, 'lr': feature_learning_rate },
            { 'params': self.sh_high_order, 'lr': feature_learning_rate / 20.0 }
        ]
        self.optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)

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
            dummy_tensor = torch.zeros_like(self.means, device="cuda", requires_grad=True)
            image, radii = rasterizer(
                self.means,
                dummy_tensor,
                self.opacities,
                None,
                self.precompute_colors(camera),
                self.scales,
                self.rotations,
                None
            )
            batch_images.append(image)
            batch_radii.append(radii)
        return torch.stack(batch_images), torch.stack(batch_radii)
    
    def get_shs(self):
        return torch.cat([self.sh_direct_current, self.sh_high_order], dim=1)
    
    def precompute_colors(self, camera: StaticCamera) -> torch.Tensor:
        """
        Precompute the colors for the given camera.

        Args:
            camera: StaticCamera

        Returns:
            torch.Tensor: Precomputed colors [..., 3]
        """
        viewing_directions = self.means - camera.camera_center
        normalized_viewing_directions = viewing_directions / torch.norm(viewing_directions, dim=1, keepdim=True)
        precomputed_colors = eval_sh(self.active_sh_degree, self.get_shs().permute(0,2,1), normalized_viewing_directions)
        precomputed_colors += 0.5
        precomputed_colors = torch.clamp_min(precomputed_colors, 0.0)
        return precomputed_colors
    
    def increment_sh_degree(self):
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