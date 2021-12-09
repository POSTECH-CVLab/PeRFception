import numpy as np

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
import dataloader.data_util.llff as llff

import jaxnerf_torch.utils as jaxnerf_torch_utils
import pytorch_lightning as pl

class ImageBatchSampler:

    def __init__(self, batch_size, image_len, image_num, seed, drop_last=False):
        self.sampler = data.sampler.SubsetRandomSampler(
            torch.arange(image_len)
        )
        self.batch_size = batch_size
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
        
        images, poses, bds, i_test = llff.load_llff_data(
            args.datadir, args.factor, recenter=True, bd_factor=0.75
        )
        hwf = poses[0, :3, -1]
        extrinsics = poses[:, :3, :4]
        h, w, focal = hwf
        h, w = int(h), int(w)
        hwf = [h, w, focal]
        self.intrinsics = np.array(
            [[focal, 0., 0.5*w], [0., focal, 0.5*h], [0., 0., 1.]]
        )

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            i_test = np.arange(images.shape[0])[:: args.llffhold]

        i_val = i_test
        is_train = lambda i : i not in i_test and i not in i_val
        i_train = np.array(
            [i for i in np.arange(len(images)) if is_train(i)]
        )
        self.near = np.ndarray.min(bds) * 0.9 if args.no_ndc else 0.
        self.far = np.ndarray.max(bds) * 1.0 if args.no_ndc else 1.

        self.image_len = h * w
        self.h, self.w = h, w

        self.train_dset = self.split(images, extrinsics, i_train)
        self.val_dset = self.split(images, extrinsics, i_val)
        self.test_dset = self.split(images, extrinsics, i_test)
        self.i_train, self.i_val, self.i_test = i_train, i_val, i_test
        
    def split(self, images, extrinsics, idx):
        images_idx = images[idx].reshape(-1, 3)
        extrinsics_idx = extrinsics[idx]
        rays_o, rays_d = jaxnerf_torch_utils.batchfied_get_rays(
            self.h, self.w, self.intrinsics, extrinsics_idx, 
            self.args.use_pixel_centers
        )
        rays = np.stack([rays_o, rays_d], axis=1)
        return LLFFImageRaySet(images_idx, rays)
    
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
            self.val_dset, batch_size=self.chunk, num_workers=12, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset, batch_size=self.chunk, num_workers=12, pin_memory=True
        )

    def get_info(self):
        return {"h": self.h, "w": self.w, "intrinsics": self.intrinsics}