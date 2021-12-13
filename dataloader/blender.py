import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler
import dataloader.data_util.blender as blender

import jaxnerf_torch.utils as jaxnerf_torch_utils
import pytorch_lightning as pl

class ImageBatchSampler(BatchSampler):

    def __init__(self, batch_size, image_len, image_num, seed):
        super(ImageBatchSampler, self).__init__(
            SubsetRandomSampler(torch.arange(image_len)), batch_size, True
        )
        self.idx_arr = np.arange(image_num)
        self.image_len = image_len
        self.rng = np.random.default_rng(seed)

    def __iter__(self): 
        image_idx = self.rng.choice(self.idx_arr, 1)[0]
        batch = []
        for idx in self.sampler:
            batch.append(idx) 
            if len(batch) == self.batch_size:
                yield batch + image_idx * self.image_len
                batch = []


class BlenderImageRaySet(Dataset):

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

        self.train_dset, _ = self.split(images, extrinsics, self.i_train)
        self.val_dset, self.val_dummy = self.split(images, extrinsics, self.i_val)
        self.test_dset, self.test_dummy = self.split(images, extrinsics, self.i_test)

    def split(self, _images, extrinsics, idx):
        images_idx = _images[idx].reshape(-1, 3)
        extrinsics_idx = extrinsics[idx]
        rays_o, rays_d = jaxnerf_torch_utils.batchfied_get_rays(
            self.h, self.w, self.intrinsics, extrinsics_idx, 
            self.args.use_pixel_centers
        )
        _rays = np.stack([rays_o, rays_d], axis=1)
        device_count = torch.cuda.device_count()
        n = len(images_idx)
        dummy_num = (device_count - len(images_idx) % device_count) % device_count
        images = np.zeros((len(images_idx) + dummy_num, 3))
        rays = np.zeros((len(images_idx) + dummy_num, 2, 3))
        images[:n], rays[:n] = images_idx, _rays
        images[n:], rays[n:] = images[:dummy_num], rays[:dummy_num]  
        
        return BlenderImageRaySet(images_idx, rays), dummy_num

    def train_dataloader(self):

        if self.args.no_batching:
            return DataLoader(
                self.train_dset, batch_sampler=ImageBatchSampler(
                    self.batch_size, self.image_len, len(self.i_train), self.seed
                ), num_workers=12, pin_memory=True
            )
        else:
            return DataLoader(
                self.train_dset, batch_size=self.batch_size, num_workers=12,
                shuffle=True, pin_memory=True
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
            "val_dummy": self.val_dummy, "test_dummy": self.test_dummy
        }