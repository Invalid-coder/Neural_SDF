import pdb

import torch
import polyscope as ps

from lib.torchgp import load_obj

class PsDebugger:
    def __init__(self):
        ps.init()
        self.pcls = {}

    def register_point_cloud(self, name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[name] = ps.register_point_cloud(name, tensor.reshape(-1, 3).numpy(), **kwargs)

    def add_vector_quantity(self, pcl_name, vec_name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[pcl_name].add_vector_quantity(vec_name, tensor.reshape(-1, 3).numpy(), **kwargs)
    
    def add_scalar_quantity(self, pcl_name, s_name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[pcl_name].add_scalar_quantity(s_name, tensor.reshape(-1).numpy(), **kwargs)
    
    def add_color_quantity(self, pcl_name, c_name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[pcl_name].add_color_quantity(c_name, tensor.reshape(-1, 3).numpy(), **kwargs)

    def add_surface_mesh(self, name, obj_path, **kwargs):
        verts, faces = load_obj(obj_path)
        ps.register_surface_mesh(name, verts.numpy(), faces.numpy(), **kwargs)
    
    def show(self):
        ps.show()
        pdb.set_trace()

