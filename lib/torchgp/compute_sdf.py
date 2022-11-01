import torch
import numpy as np
import mesh2sdf

def compute_sdf(
    V : torch.Tensor,
    F : torch.Tensor,
    points : torch.Tensor):
    """Given a [N,3] list of points, returns a [N] list of SDFs for a mesh."""

    mesh = V[F]

    points_cpu = points.cpu().numpy().reshape(-1).astype(np.float64)
    mesh_cpu = mesh.cpu().numpy().reshape(-1).astype(np.float64)

    # Legacy, open source mesh2sdf code
    dist = mesh2sdf.mesh2sdf_gpu(points.contiguous(), mesh)[0]
    
    return dist
