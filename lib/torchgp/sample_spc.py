import torch


def sample_spc(
    corners     : torch.Tensor, 
    level       : int,
    num_samples : int):
    """Sample uniformly in [-1,1] bounding volume within SPC voxels
    
    Args:
        corners (tensor)  : set of corners to sample from
        level (int)       : level to sample from 
        num_samples (int) : number of points to sample
    """

    res = 2.0**level
    samples = torch.rand(corners.shape[0], num_samples, 3, device=corners.device)
    samples = corners[...,None,:3] + samples
    samples = samples.reshape(-1, 3)
    samples /= res
    return samples * 2.0 - 1.0

