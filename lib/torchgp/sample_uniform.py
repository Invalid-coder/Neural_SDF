import torch

def sample_uniform(num_samples : int):
    """Sample uniformly in [-1,1] bounding volume.
    
    Args:
        num_samples(int) : number of points to sample
    """
    return torch.rand(num_samples, 3) * 2.0 - 1.0

