import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.Embedder import positional_encoding
from lib.models.BasicDecoder import BasicDecoder
from lib.utils import setparam

class BaseSDF(nn.Module):
    def __init__(self,
        args             = None,
        pos_enc  : bool  = None,
        ff_dim   : int   = None,
        ff_width : float = None
    ):
        super().__init__()
        self.args = args
        self.pos_enc = setparam(args, pos_enc, 'pos_enc')
        self.ff_dim = setparam(args, ff_dim, 'ff_dim')
        self.ff_width = setparam(args, ff_width, 'ff_width')
        
        self.input_dim = 3
        self.out_dim = 1

        if self.ff_dim > 0:
            mat = torch.randn([self.ff_dim, 3]) * self.ff_width
            self.gauss_matrix = nn.Parameter(mat)
            self.gauss_matrix.requires_grad_(False)
            self.input_dim += (self.ff_dim * 2) - 3
        elif self.pos_enc:
            self.input_dim = self.input_dim * 13

    def forward(self, x, lod=None):
        x = self.encode(x)
        return self.sdf(x)
 
    def freeze(self):
        for k, v in self.named_parameters():
            v.requires_grad_(False)

    def encode(self, x):
        if self.ff_dim > 0:
            x = F.linear(x, self.gauss_matrix)
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        elif self.pos_enc:
            x = positional_encoding(x)
        return x

    def sdf(self, x, lod=None):
        return None
