# Virtual Normal Loss (VNL)

A highly optimized PyTorch implementation of Virtual Normal Loss for metric depth estimation.

Traditional depth estimation metrics compute pixel-wise differences (2D). VNL leverages camera intrinsics to reconstruct the 3D point cloud, randomly samples coordinate triplets to form planes, and computes the angular divergence between predicted and ground-truth 3D surface normals.

# Performance
This library computes VNL natively on the GPU without Python loops. By utilizing PyTorch meshgrids for 3D projection and vectorized `torch.linalg.cross` operations, it evaluates thousands of geometric constraints in milliseconds.

# Installation

```bash
git clone  https://github.com/dhyey004/vnl_loss.git
cd vnl_loss
pip install -e .
