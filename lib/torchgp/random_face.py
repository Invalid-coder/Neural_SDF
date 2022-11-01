import torch
from .area_weighted_distribution import area_weighted_distribution
from .per_face_normals import per_face_normals

def random_face(
    V : torch.Tensor, 
    F : torch.Tensor, 
    num_samples : int, 
    distrib=None):
    """Return an area weighted random sample of faces and their normals from the mesh.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        num_samples (int): num of samples to return
        distrib: distribution to use. By default, area-weighted distribution is used.
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)

    normals = per_face_normals(V, F)

    idx = distrib.sample([num_samples])

    return F[idx], normals[idx]

