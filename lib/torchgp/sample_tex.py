import torch
import torch.nn.functional as F

def sample_tex(
    Tp : torch.Tensor, # points [N ,2] 
    TM : torch.Tensor, # material indices [N]
    materials):

    max_idx = TM.max()
    assert(max_idx > -1 and "No materials detected! Check the material definiton on your mesh.")

    rgb = torch.zeros(Tp.shape[0], 3, device=Tp.device)

    Tp = (Tp * 2.0) - 1.0
    # The y axis is flipped from what UV maps generally expects vs in PyTorch
    Tp[...,1] *= -1

    for i in range(max_idx+1):
        mask = (TM == i)
        if mask.sum() == 0:
            continue
        if 'diffuse_texname' not in materials[i]:
            if 'diffuse' in materials[i]:
                rgb[mask] = materials[i]['diffuse'].to(Tp.device)
            continue

        map = materials[i]['diffuse_texname'][...,:3].permute(2, 0, 1)[None].to(Tp.device)
        grid = Tp[mask]
        grid = grid.reshape(1, grid.shape[0], 1, grid.shape[1])
        _rgb = F.grid_sample(map, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        _rgb = _rgb[0,:,:,0].permute(1,0)
        rgb[mask] = _rgb

    return rgb


