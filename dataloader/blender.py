import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import dataloader.data_util.blender as blender

import model.jaxnerf_torch.utils as jaxnerf_torch_utils
import pytorch_lightning as pl

from dataloader.sampler import SingleImageSampler, SingleImageDDPSampler, RaySet

class LitBlender(pl.LightningDataModule):

    def __init__(self, args):
        self.args = args
        self.batch_size = args.N_rand
        self.seed = args.seed
        self.chunk = args.chunk

        images, poses, render_poses, hwf, i_split = blender.load_blender_data(
            args.datadir, args.half_res, args.testskip
        )
        i_train, i_val, i_test = i_split

        self.near = 2.
        self.far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

        extrinsics = poses
        h, w, focal = hwf
        h, w = int(h), int(w)
        hwf = [h, w, focal]

        self.image_len = h * w
        self.h, self.w = h, w

        self.intrinsics = np.array(
            [[focal, 0., 0.5*w], [0., focal, 0.5*h], [0., 0., 1.]]
        )

        self.i_train, self.i_val = i_train, i_val
        self.i_test = np.arange(len(images))

        self.train_dset, _ = self.split(images, extrinsics, self.i_train, False)
        self.val_dset, self.val_dummy = self.split(images, extrinsics, self.i_val)
        self.test_dset, self.test_dummy = self.split(images, extrinsics, self.i_test)

    def split(self, _images, extrinsics, idx, dummy=True):
        images_idx = _images[idx].reshape(-1, 3)
        extrinsics_idx = extrinsics[idx]
        rays_o, rays_d = jaxnerf_torch_utils.batchfied_get_rays(
            self.h, self.w, self.intrinsics, extrinsics_idx, 
            self.args.use_pixel_centers
        )
        _rays = np.stack([rays_o, rays_d], axis=1)
        device_count = torch.cuda.device_count()
        n = len(images_idx)
        if dummy:
            dummy_num = (device_count - len(images_idx) % device_count) % device_count
            images = np.zeros((len(images_idx) + dummy_num, 3))
            rays = np.zeros((len(images_idx) + dummy_num, 2, 3))
            images[:n], rays[:n] = images_idx, _rays
            images[n:], rays[n:] = images[:dummy_num], rays[:dummy_num]  
        else:
            dummy_num = 0
            images = images_idx
            rays = _rays
        
        return RaySet(images, rays), dummy_num

    def train_dataloader(self):

        assert self.args.batching == "single_image"
        sampler = SingleImageDDPSampler(
            self.batch_size, len(self.i_train), self.image_len, 
            self.args.i_validation
        ) if not self.args.tpu else BatchSampler(SingleImageSampler(
            self.batch_size, len(self.i_train), self.image_len, self.args.i_validation, self.args.tpu_num,
        ), batch_size=self.batch_size // self.args.tpu_num, drop_last=False)
        return DataLoader(
            self.train_dset, batch_sampler=sampler, num_workers=12, 
            pin_memory=True, shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset, batch_size=self.chunk, num_workers=12, 
            pin_memory=True, drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset, batch_size=self.chunk, num_workers=12, 
            pin_memory=True, drop_last=False
        )

    def get_info(self):
        return {
            "h": self.h, "w": self.w, "intrinsics": self.intrinsics,
            "i_train": self.i_train, "i_val": self.i_val, "i_test": self.i_test,
            "val_dummy": self.val_dummy, "test_dummy": self.test_dummy,
            "near": self.near, "far": self.far
        }