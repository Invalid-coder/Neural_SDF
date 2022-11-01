import torch

def per_face_normals(
    V : torch.Tensor,
    F : torch.Tensor):
    """Compute normals per face.
    """
    mesh = V[F]

    vec_a = mesh[:, 0] - mesh[:, 1]
    vec_b = mesh[:, 1] - mesh[:, 2]
    normals = torch.cross(vec_a, vec_b)
    return normals

