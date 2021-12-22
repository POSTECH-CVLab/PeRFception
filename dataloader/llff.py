import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

import dataloader.data_util.llff as llff

import model.jaxnerf_torch.utils as jaxnerf_torch_utils
import pytorch_lightning as pl

from dataloader.sampler import RaySet, MultipleImageDDPSampler, DDPSequnetialSampler
from dataloader.interface import LitData

class LitLLFF(LitData):

    def __init__(self, args):
        super(LitLLFF, self).__init__(args)

        images, poses, bds, render_poses, i_test = llff.load_llff_data(
            args.datadir, args.factor, recenter=True, bd_factor=0.75
        )
        hwf = poses[0, :3, -1]
        extrinsics = poses[:, :3, :4]
        h, w, focal = hwf
        h, w = int(h), int(w)
        hwf = [h, w, focal]
        self.intrinsics = np.array([[focal, 0., 0.5 * w], [0., focal, 0.5 * h],
                                    [0., 0., 1.]])

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        is_train = lambda i: i not in i_test and i not in i_val
        i_train = np.array([i for i in np.arange(len(images)) if is_train(i)])
        self.near = np.ndarray.min(bds) * 0.9 if args.no_ndc else 0.
        self.far = np.ndarray.max(bds) * 1.0 if args.no_ndc else 1.

        self.image_len = h * w
        self.h, self.w = h, w

        self.i_train, self.i_val, self.i_test = i_train, i_val, i_test
        self.i_all = np.arange(len(images))
        self.train_dset, _ = self.split(images, extrinsics, self.i_train, False)
        self.val_dset, self.val_dummy = self.split(images, extrinsics, self.i_val)
        self.test_dset, self.test_dummy = self.split(images, extrinsics, self.i_all)

        render_poses = np.stack(render_poses)[..., :4]
        N_render = len(render_poses)
        self.predict_dset, self.pred_dummy = self.split(None, render_poses, np.arange(N_render))

    def train_dataloader(self):

        if self.args.tpu:
            import torch_xla.core.xla_model as xm
            sampler = MultipleImageDDPSampler(
                self.batch_size, xm.xrt_world_size(), xm.get_ordinal(),
                len(self.train_dset), self.args.i_validation
            )
        else:
            sampler = MultipleImageDDPSampler(
                self.batch_size, None, None, len(self.train_dset), self.args.i_validation
            )

        return DataLoader(
            self.train_dset, batch_sampler=sampler, num_workers=self.num_workers,
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
            self.predict_dset, batch_size=self.chunk, sampler=sampler,
            num_workers=self.args.num_workers, pin_memory=True, shuffle=False
        )
