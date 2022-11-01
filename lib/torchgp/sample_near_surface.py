import torch
from .sample_surface import sample_surface
from .area_weighted_distribution import area_weighted_distribution

def sample_near_surface(
    V : torch.Tensor,
    F : torch.Tensor, 
    num_samples: int, 
    variance : float = 0.01,
    distrib=None):
    """Sample points near the mesh surface.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        num_samples (int): number of surface samples
        distrib: distribution to use. By default, area-weighted distribution is used
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)
    samples = sample_surface(V, F, num_samples, distrib)[0]
    samples += torch.randn_like(samples) * variance
    return samples
