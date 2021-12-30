import numpy as np

import torch
from torch.utils.data import DataLoader

import dataloader.data_util.blender as blender

from dataloader.sampler import SingleImageDDPSampler, DDPSequnetialSampler
from dataloader.interface import LitData

class LitBlender(LitData):
    
    def __init__(self, args):
        super(LitBlender, self).__init__(args)

        images, poses, render_poses, hwf, i_split = blender.load_blender_data(
            args.datadir, args.half_res, args.testskip)
        i_train, i_val, i_test = i_split

        self.near = 2.
        self.far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

        extrinsics = poses
        h, w, focal = hwf
        h, w = int(h), int(w)
        hwf = [h, w, focal]

        self.image_len = h * w
        self.h, self.w = h, w

        self.intrinsics = np.array(
            [[focal, 0., 0.5 * w], [0., focal, 0.5 * h], [0., 0., 1.]]
        )
        self.extrinsics = poses

        self.i_train, self.i_val, self.i_test = i_train, i_val, i_test
        self.i_all = np.arange(len(images))

        self.train_dset, _ = self.split(images, extrinsics, self.i_train, False)
        self.val_dset, self.val_dummy = self.split(images, extrinsics, self.i_val)
        self.test_dset, self.test_dummy = self.split(images, extrinsics, self.i_all)

        N_render = len(render_poses)
        self.predict_dset, self.pred_dummy = self.split(None, render_poses, np.arange(N_render))


    def train_dataloader(self):

        assert self.args.batching == "single_image"

        if self.args.tpu:
            import torch_xla.core.xla_model as xm
            sampler = SingleImageDDPSampler(
                self.batch_size, xm.xrt_world_size(), xm.get_ordinal(),
                len(self.i_train), self.image_len, self.args.i_validation
            )
        else:
            sampler = SingleImageDDPSampler(
                self.batch_size, None, None, len(self.i_train), 
                self.image_len, self.args.i_validation
            )
        return DataLoader(
            self.train_dset, batch_sampler=sampler, num_workers=self.args.num_workers,
            pin_memory=True, shuffle=False
        )

    def val_dataloader(self):
        
        if self.args.tpu: 
            import torch_xla.core.xla_model as xm
            sampler = DDPSequnetialSampler(self.chunk, xm.xrt_world_size(), xm.get_ordinal(), len(self.val_dset))
        else:
            sampler = DDPSequnetialSampler(self.chunk, None, None, len(self.val_dset))
        
        return DataLoader(
            self.val_dset, batch_size=self.chunk, sampler=sampler, 
            num_workers=self.args.num_workers, pin_memory=True, shuffle=False
        )

    def test_dataloader(self):
        
        if self.args.tpu: 
            import torch_xla.core.xla_model as xm
            sampler = DDPSequnetialSampler(self.chunk, xm.xrt_world_size(), xm.get_ordinal(), len(self.test_dset))
        else:
            sampler = DDPSequnetialSampler(self.chunk, None, None, len(self.test_dset))
        
        return DataLoader(
            self.test_dset, batch_size=self.chunk, sampler=sampler,
            num_workers=self.args.num_workers, pin_memory=True, shuffle=False
        )

    def predict_dataloader(self):
        if self.args.tpu: 
            import torch_xla.core.xla_model as xm
            sampler = DDPSequnetialSampler(self.chunk, xm.xrt_world_size(), xm.get_ordinal(), len(self.predict_dset))
        else:
            sampler = DDPSequnetialSampler(self.chunk, None, None, len(self.predict_dset))
        
        return DataLoader(
            self.predict_dset, batch_size=self.args.chunk, sampler=sampler,
            num_workers=self.args.num_workers, pin_memory=True, shuffle=False
        )
