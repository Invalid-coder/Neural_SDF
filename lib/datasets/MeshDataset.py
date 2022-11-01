import os

import torch
from torch.utils.data import Dataset

import numpy as np
import mesh2sdf

from lib.torchgp import load_obj, point_sample, sample_surface, compute_sdf, normalize
from lib.PsDebugger import PsDebugger

from lib.utils import PerfTimer, setparam

class MeshDataset(Dataset):
    """Base class for single mesh datasets."""

    def __init__(self, 
        args=None, 
        dataset_path = None,
        raw_obj_path = None,
        sample_mode = None,
        get_normals = None,
        seed = None,
        num_samples = None,
        trim = None,
        sample_tex = None
    ):
        self.args = args
        self.dataset_path = setparam(args, dataset_path, 'dataset_path')
        self.raw_obj_path = setparam(args, raw_obj_path, 'raw_obj_path')
        self.sample_mode = setparam(args, sample_mode, 'sample_mode')
        self.get_normals = setparam(args, get_normals, 'get_normals')
        self.num_samples = setparam(args, num_samples, 'num_samples')
        self.trim = setparam(args, trim, 'trim')
        self.sample_tex = setparam(args, sample_tex, 'sample_tex')

        if self.sample_tex:
            out = load_obj(self.dataset_path, load_materials=True)
            self.V, self.F, self.texv, self.texf, self.mats = out
        else:
            self.V, self.F = load_obj(self.dataset_path)

        self.V, self.F = normalize(self.V, self.F)
        self.mesh = self.V[self.F]
        self.resample()

    def resample(self):
        """Resample SDF samples."""

        self.nrm = None
        if self.get_normals:
            self.pts, self.nrm = sample_surface(self.V, self.F, self.num_samples*5)
            self.nrm = self.nrm.cpu()
        else:
            self.pts = point_sample(self.V, self.F, self.sample_mode, self.num_samples)

        self.d = compute_sdf(self.V.cuda(), self.F.cuda(), self.pts.cuda())   

        self.d = self.d[...,None]
        self.d = self.d.cpu()
        self.pts = self.pts.cpu()

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.get_normals:
            return self.pts[idx], self.d[idx], self.nrm[idx]
        elif self.sample_tex:
            return self.pts[idx], self.d[idx], self.rgb[idx]
        else:
            return self.pts[idx], self.d[idx]
            
    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.pts.size()[0]

    def num_shapes(self):
        """Return length of dataset (number of _mesh models_)."""

        return 1
