import torch
import torch.nn as nn
import torch.nn.functional as F

class VirtualNormalLoss(nn.Module):
    def __init__(self, focal_x: float, focal_y: float, center_x: float, center_y: float, num_samples: int = 3000):
        super().__init__()
        self.fx = focal_x
        self.fy = focal_y
        self.cx = center_x
        self.cy = center_y
        self.num_samples = num_samples

    def depth_to_3d(self, depth: torch.Tensor) -> torch.Tensor:
        """Converts a depth map (B, 1, H, W) to a 3D point cloud (B, H*W, 3)."""
        B, C, H, W = depth.shape
        
        # Create a meshgrid of pixel coordinates
        y, x = torch.meshgrid(torch.arange(H, device=depth.device), 
                              torch.arange(W, device=depth.device), indexing='ij')
        
        # Reconstruct 3D points
        Z = depth.reshape(B, -1)
        X = (x.reshape(-1) - self.cx) * Z / self.fx
        Y = (y.reshape(-1) - self.cy) * Z / self.fy
        
        # Stack into (B, H*W, 3)
        points_3d = torch.stack([X, Y, Z], dim=-1)
        return points_3d

    def compute_normals(self, points: torch.Tensor, idx1: torch.Tensor, idx2: torch.Tensor, idx3: torch.Tensor) -> torch.Tensor:
        """Computes normal vectors for triangles formed by sampled points."""
        p1 = points[:, idx1, :]
        p2 = points[:, idx2, :]
        p3 = points[:, idx3, :]

        # Create vectors from points
        v1 = p2 - p1
        v2 = p3 - p1

        # Cross product to find normal, then normalize
        normals = torch.linalg.cross(v1, v2, dim=-1)
        normals = F.normalize(normals, p=2, dim=-1)
        return normals

    def forward(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred_depth.shape
        num_pixels = H * W

        # 1. Project depth to 3D point clouds
        pred_3d = self.depth_to_3d(pred_depth)
        gt_3d = self.depth_to_3d(gt_depth)

        # 2. Randomly sample triplets of points
        idx1 = torch.randint(0, num_pixels, (self.num_samples,), device=pred_depth.device)
        idx2 = torch.randint(0, num_pixels, (self.num_samples,), device=pred_depth.device)
        idx3 = torch.randint(0, num_pixels, (self.num_samples,), device=pred_depth.device)

        # 3. Compute normals for both prediction and ground truth
        pred_normals = self.compute_normals(pred_3d, idx1, idx2, idx3)
        gt_normals = self.compute_normals(gt_3d, idx1, idx2, idx3)

        # 4. Compute L1 Loss between the normals
        loss = torch.mean(torch.abs(pred_normals - gt_normals))
        return loss