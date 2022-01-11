import pytorch_lightning as pl

import model.jaxnerf_torch.utils as jaxnerf_torch_utils
from dataloader.sampler import RaySet
from dataloader.data_util.ray import batchfied_get_rays

import numpy as np
import torch
import cv2

class LitData(pl.LightningDataModule):

    def __init__(self, args): 
        super(LitData, self).__init__()
        self.args = args
        self.batch_size = args.N_rand
        self.seed = args.seed
        self.chunk = args.chunk
        self.num_workers = args.num_workers

    def check(self):

        assert hasattr(self, "h"), "You must define self.h"
        assert hasattr(self, "w"), "You must define self.w"
        assert hasattr(self, "near"), "You must define self.near"
        assert hasattr(self, "far"), "You must define self.far"
        assert hasattr(self, "i_train"), "You must define self.i_train"
        assert hasattr(self, "i_val"), "You must define self.i_val"
        assert hasattr(self, "i_test"), "You must define self.i_test"

    def convert_to_ndc(
        self,
        origins,
        directions,
        ndc_coeffs,
        near: float = 1.0
    ):
        """Convert a set of rays to NDC coordinates."""
        # Shift ray origins to near plane, not sure if needed
        # Projection
        t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
        origins = origins + t[Ellipsis, None] * directions

        dx, dy, dz = directions[:, 0], directions[:, 1], directions[:, 2]
        ox, oy, oz = origins[:, 0], origins[:, 1], origins[:, 2]
        o0 = ndc_coeffs[0] * (ox / oz)
        o1 = ndc_coeffs[1] * (oy / oz)
        o2 = 1 - 2 * near / oz
        d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
        d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
        d2 = 2 * near / oz

        origins = np.stack([o0, o1, o2], -1)
        directions = np.stack([d0, d1, d2], -1)

        return origins, directions

    def split(self, _images, extrinsics, idx, dummy=True):
        # dummy is needed for DDP evaluation but not necessary for the training phase
        image_exist = _images is not None
        images = None
        
        H, W = self.h, self.w
        intrinsics = self.intrinsics
        ndc_coeffs = (W / (2.0 * intrinsics[0][0]), H / (2.0 * intrinsics[1][1]))

        extrinsics_idx = extrinsics[idx]
        rays_o, rays_d = batchfied_get_rays(
            H, W, intrinsics, extrinsics_idx, self.args.use_pixel_centers
        )
        if not self.args.no_ndc:
            rays_o, rays_d = self.convert_to_ndc(rays_o, rays_d, ndc_coeffs)
            rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)

        _rays = np.stack([rays_o, rays_d], axis=1)
        device_count = torch.cuda.device_count() if not self.args.tpu else self.args.tpu_num
        n_dset = len(_rays)

        dummy_num = (device_count - n_dset % device_count) % device_count if dummy else 0
        rays = np.zeros((n_dset + dummy_num, 2, 3))
        rays[:n_dset] = _rays
        if dummy:
            rays[n_dset:] = rays[:dummy_num]

        if image_exist:
            images_idx = _images[idx].reshape(-1, 3)
            images = np.zeros((n_dset + dummy_num, 3))
            images[:n_dset] = images_idx 
            images[n_dset:] = images[:dummy_num]

        return RaySet(images, rays), dummy_num