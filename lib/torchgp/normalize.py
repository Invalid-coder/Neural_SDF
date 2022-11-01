import torch

def normalize(
    V : torch.Tensor,
    F : torch.Tensor):

    # Normalize mesh
    V_max, _ = torch.max(V, dim=0)
    V_min, _ = torch.min(V, dim=0)
    V_center = (V_max + V_min) / 2.
    V = V - V_center

    # Find the max distance to origin
    max_dist = torch.sqrt(torch.max(torch.sum(V**2, dim=-1)))
    V_scale = 1. / max_dist
    V *= V_scale
    return V, F
