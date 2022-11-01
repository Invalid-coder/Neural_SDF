import torch
from .per_face_normals import per_face_normals

def area_weighted_distribution(
    V : torch.Tensor,
    F : torch.Tensor, 
    normals : torch.Tensor = None):
    """Construct discrete area weighted distribution over triangle mesh.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        normals (torch.Tensor): normals (if precomputed)
        eps (float): epsilon
    """

    if normals is None:
        normals = per_face_normals(V, F)
    areas = torch.norm(normals, p=2, dim=1) * 0.5
    areas /= torch.sum(areas) + 1e-10
    
    # Discrete PDF over triangles
    return torch.distributions.Categorical(areas.view(-1))

