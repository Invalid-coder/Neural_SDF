import os
import sys
import itertools as it

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lib.datasets import *
from lib.validator.metrics import *

class GeometricValidator(object):
    """Geometric validation; sample 3D points for distance/occupancy metrics."""

    def __init__(self, args, device, net):
        self.args = args
        self.device = device
        self.net = net
        self.num_samples = 100000
        self.set_dataset()

    
    def set_dataset(self):
        """Two datasets; 1) samples uniformly for volumetric IoU, 2) samples surfaces only."""

        # Same as training since we're overfitting
        self.val_dataset = MeshDataset(self.args, num_samples=self.num_samples)
        self.val_data_loader = DataLoader(self.val_dataset, 
                                          batch_size=self.num_samples*len(self.args.sample_mode),
                                          shuffle=False, pin_memory=True, num_workers=4)


    def validate(self, epoch):
        """Geometric validation; sample surface points."""

        val_dict = {}
        val_dict['vol_iou'] = []
        
        # Uniform points metrics
        for n_iter, data in enumerate(self.val_data_loader):

            ids = data[0].to(self.device)
            pts = data[1].to(self.device)
            gts = data[2].to(self.device)
            nrm = data[3].to(self.device) if self.args.get_normals else None

            for d in range(self.args.num_lods):
                self.net.lod = d

                # Volumetric IoU
                pred = self.net(pts, gts=gts, grad=nrm, ids=ids)
                val_dict['vol_iou'] += [float(compute_iou(gts, pred))]
                self.net.lod = None

        return val_dict

