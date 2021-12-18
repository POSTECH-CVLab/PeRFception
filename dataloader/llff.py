import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

import dataloader.data_util.llff as llff

import model.jaxnerf_torch.utils as jaxnerf_torch_utils
import pytorch_lightning as pl

from dataloader.sampler import RaySet, MultipleImageDDPSampler, DDPSequnetialSampler


class LLFFImageRaySet(data.Dataset):
    def __init__(self, images, rays):
        self.images = images
        self.rays = rays
        self.N = len(images)

    def __getitem__(self, index):
        return {
            "target": torch.from_numpy(self.images[index]),
            "ray": torch.from_numpy(self.rays[index])
        }

    def __len__(self):
        return len(self.images)


class LitLLFF(pl.LightningDataModule):
    def __init__(self, args):
        self.args = args
        self.batch_size = args.N_rand
        self.seed = args.seed
        self.chunk = args.chunk

        images, poses, bds, i_test = llff.load_llff_data(args.datadir,
                                                         args.factor,
                                                         recenter=True,
                                                         bd_factor=0.75)
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
            self.args.use_pixel_centers)
        _rays = np.stack([rays_o, rays_d], axis=1)
        device_count = torch.cuda.device_count(
        ) if not self.args.tpu else self.args.tpu_num
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

        if self.args.tpu:
            import torch_xla.core.xla_model as xm
            sampler = MultipleImageDDPSampler(
                self.batch_size, xm.xrt_world_size(), xm.get_ordinal(),
                len(self.train_dset), self.i_validation
            )
        else:
            sampler = MultipleImageDDPSampler(
                self.batch_size, None, None, len(self.train_dset), self.args.i_validation
            )

        return DataLoader(
            self.train_dset, batch_sampler=sampler, num_workers=self.args.num_workers,
            pin_memory=True, shuffle=False
        )

    def val_dataloader(self):
        
        if self.args.tpu: 
            import torch_xla.core.xla_model as xm
            sampler = DDPSequnetialSampler(self.args.chunk, xm.xrt_world_size(), xm.get_ordinal(), len(self.val_dset))
        else:
            sampler = DDPSequnetialSampler(self.args.chunk, None, None, len(self.val_dset))
        
        return DataLoader(
            self.val_dset, batch_size=self.args.chunk, sampler=sampler, 
            num_workers=self.args.num_workers, pin_memory=True, shuffle=False
        )

    def test_dataloader(self):
        
        if self.args.tpu: 
            import torch_xla.core.xla_model as xm
            sampler = DDPSequnetialSampler(self.args.chunk, xm.xrt_world_size(), xm.get_ordinal(), len(self.test_dset))
        else:
            sampler = DDPSequnetialSampler(self.args.chunk, None, None, len(self.test_dset))
        
        return DataLoader(
            self.test_dset, batch_size=self.args.chunk, sampler=sampler, 
            num_workers=self.args.num_workers, pin_memory=True, shuffle=False
        )