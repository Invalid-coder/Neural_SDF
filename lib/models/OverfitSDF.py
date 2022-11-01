import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.BaseSDF import BaseSDF
from lib.models.BasicDecoder import BasicDecoder

class OverfitSDF(BaseSDF):
    def __init__(self, args):
        super().__init__(args)
        
        self.decoder = BasicDecoder(self.input_dim, self.out_dim, F.relu, True, self.args)

    def sdf(self, x, lod=None):
        return self.decoder(x)

