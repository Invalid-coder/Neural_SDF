import numpy as np
import logging as log

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.BaseSDF import BaseSDF

class BaseLOD(BaseSDF):
    def __init__(self, args):
        super().__init__(args)
        self.num_lods = args.num_lods
        self.lod = None

    def forward(self, x, lod=None):
        if lod is None:
            lod = self.lod
        x = self.encode(x)
        return self.sdf(x)

    def sdf(self, x, lod=None):
        if lod is None:
            lod = self.lod
        return None


