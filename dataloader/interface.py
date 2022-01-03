import pytorch_lightning as pl

import model.jaxnerf_torch.utils as jaxnerf_torch_utils
from dataloader.sampler import RaySet

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

    
    def split(self, _images, extrinsics, idx, dummy=True, scene_scale=1.0):
        # dummy is needed for DDP evaluation but not necessary for the training phase
        image_exist = _images is not None
        images = None
        
        H, W = self.h, self.w
        intrinsics = self.intrinsics

        if scene_scale != 1.0:
            H, W = int(self.h * scene_scale), int(self.w * scene_scale) 
            intrinsics = self.intrinsics.copy()
            intrinsics[[0, 0, 1, 1], [0, 2, 1, 2]] *= scene_scale
            extrinsics[:, :3, 3] *= scene_scale

            imgs_scaled_res = np.zeros((len(idx), H, W, 3))
            for i, img in enumerate(_images[idx]):
                imgs_scaled_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            _images = imgs_scaled_res

        extrinsics_idx = extrinsics[idx]
        rays_o, rays_d = jaxnerf_torch_utils.batchfied_get_rays(
            H, W, intrinsics, extrinsics_idx, self.args.use_pixel_centers,
        )
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