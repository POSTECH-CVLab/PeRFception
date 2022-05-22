import pytorch_lightning as pl

from dataloader.sampler import RaySet
from utils.ray import batchified_get_rays

import numpy as np
from torch.utils.data import DataLoader
from dataloader.sampler import (
    SingleImageDDPSampler, DDPSequnetialSampler, 
    MultipleImageDDPSampler, MultipleImageWOReplaceDDPSampler
)
from typing import *
import gin

@gin.configurable()
class LitData(pl.LightningDataModule):

    def __init__(
        self,
        datadir: str, 
        accelerator: str,
        num_gpus: Optional[int],
        num_tpus: Optional[int],
        batch_size: int = 4096,
        chunk: int = 1024 * 32, 
        num_workers: int = 4,
        ndc_coord: bool = False,
        batch_sampler: str = "all_images_wo_replace",
        eval_test_only: bool = True,
        epoch_size: int = 50000,
        use_pixel_centers: bool = True,
        white_bkgd: bool = False,
        precrop: bool = False,
        precrop_steps: int = 0, 
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        self.use_tpu = accelerator == "tpu"

        self.scene_center = [0., 0., 0.]
        self.scene_radius = [1., 1., 1.]
        self.use_sphere_bound = True

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dset, _ = self.split_each(
                self.images, None, self.i_train, False
            )
            self.val_dset, self.val_dummy = self.split_each(
                self.images, None, self.i_val, 
            )
            self.train_image_sizes = self.image_sizes[self.i_train]
            self.val_image_sizes = self.image_sizes[self.i_val]
        
        if stage == "test" or stage is None:
            test_idx = self.i_test if self.eval_test_only else self.i_all
            self.test_dset, self.test_dummy = self.split_each(
                self.images, None, test_idx, 
            )
            self.test_image_sizes = self.image_sizes[self.i_test]
            self.all_image_sizes = self.image_sizes[self.i_all]
        
        if stage == "predict" or stage is None:
            render_poses = np.stack(self.render_poses)
            self.predict_dset, self.pred_dummy = self.split_each(
                None, render_poses, np.arange(len(render_poses))
            )

    def split_each(
        self, 
        _images, 
        render_poses,
        idx, 
        dummy=True
    ):
        image_exist = _images is not None
        images = None

        if image_exist:
            extrinsics_idx = self.extrinsics[idx]
            intrinsics_idx = self.intrinsics[idx]
            image_sizes_idx = self.image_sizes[idx]
        else:
            extrinsics_idx = render_poses
            N_render = len(render_poses)
            intrinsics_idx = np.stack(
                [self.intrinsics[0] for _ in range(N_render)]
            )
            image_sizes_idx = np.stack(
                [self.image_sizes[0] for _ in range(N_render)]
            )

            # Only when image is rescaled
            if hasattr(self, "render_scale"): 
                intrinsics_idx[:, [0, 0, 1, 1], [0, 2, 1, 2]] *= self.render_scale
                image_sizes_idx = (image_sizes_idx * self.render_scale).astype(image_sizes_idx.dtype)
            
        rays_o, rays_d = batchified_get_rays(
            intrinsics_idx, 
            extrinsics_idx, 
            image_sizes_idx,
            self.use_pixel_centers
        )

        _rays = np.stack([rays_o, rays_d], axis=1)
        device_count = self.num_gpus if not self.use_tpu else self.num_tpus
        n_dset = len(_rays)

        dummy_num = (
            device_count - n_dset % device_count
        ) % device_count if dummy else 0

        rays = np.zeros((n_dset + dummy_num, 2, 3), dtype=np.float32)
        rays[:n_dset] = _rays

        if dummy:
            rays[n_dset:] = rays[:dummy_num]

        if image_exist:
            images_idx = np.concatenate([_images[i].reshape(-1, 3) for i in idx])
            images = np.zeros((n_dset + dummy_num, 3))
            images[:n_dset] = images_idx
            images[n_dset:] = images[:dummy_num]

        return RaySet(images, rays), dummy_num

    
    def train_dataloader(self):

        if self.batch_sampler == "single_image":
            if self.use_tpu:
                import torch_xla.core.xla_model as xm
                sampler = SingleImageDDPSampler(
                    batch_size=self.batch_size, 
                    num_replicas=xm.xrt_world_size(), 
                    rank=xm.get_ordinal(),
                    N_img=len(self.i_train), 
                    N_pixels=self.image_len, 
                    epoch_size=self.epoch_size, 
                    tpu=True, 
                    precrop=self.precrop,
                    precrop_steps=self.precrop_steps
                )
            else:
                sampler = SingleImageDDPSampler(
                    batch_size=self.batch_size, 
                    num_replicas=None, 
                    rank=None, 
                    N_img=len(self.i_train), 
                    N_pixels=self.image_sizes[self.i_train], 
                    epoch_size=self.epoch_size, 
                    tpu=False,
                    precrop=self.precrop,
                    precrop_steps=self.precrop_steps
                )
        elif self.batch_sampler == "all_images":
            if self.use_tpu:
                import torch_xla.core.xla_model as xm
                sampler = MultipleImageDDPSampler(
                    batch_size=self.batch_size, 
                    num_replicas=xm.xrt_world_size(), 
                    rank=xm.get_ordinal(),
                    total_len=len(self.train_dset), 
                    epoch_size=self.epoch_size, 
                    tpu=True
                )
            else:
                sampler = MultipleImageDDPSampler(
                    batch_size=self.batch_size, 
                    num_replicas=None, 
                    rank=None, 
                    total_len=len(self.train_dset), 
                    epoch_size=self.epoch_size, 
                    tpu=False
                )

        elif self.batch_sampler == "all_images_wo_replace":
            if self.use_tpu:
                import torch_xla.core.xla_model as xm
                sampler = MultipleImageWOReplaceDDPSampler(
                    batch_size=self.batch_size, 
                    num_replicas=xm.xrt_world_size(), 
                    rank=xm.get_ordinal(),
                    total_len=len(self.train_dset), 
                    epoch_size=self.epoch_size, 
                    tpu=True
                )
            else:
                sampler = MultipleImageWOReplaceDDPSampler(
                    batch_size=self.batch_size, 
                    num_replicas=None, 
                    rank=None, 
                    total_len=len(self.train_dset), 
                    epoch_size=self.epoch_size, 
                    tpu=False
                )

        return DataLoader(
            dataset=self.train_dset, 
            batch_sampler=sampler, 
            num_workers=self.num_workers,
            pin_memory=True, 
            shuffle=False,
            persistent_workers=True,
        )

    def val_dataloader(self):
        
        if self.use_tpu: 
            import torch_xla.core.xla_model as xm
            sampler = DDPSequnetialSampler(
                batch_size=self.chunk, 
                num_replicas=xm.xrt_world_size(), 
                rank=xm.get_ordinal(), 
                N_total=len(self.val_dset), 
                tpu=True
            )
        else:
            sampler = DDPSequnetialSampler(
                batch_size=self.chunk, 
                num_replicas=None, 
                rank=None, 
                N_total=len(self.val_dset), 
                tpu=False
            )
        
        return DataLoader(
            dataset=self.val_dset, 
            batch_size=self.chunk, 
            sampler=sampler, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            shuffle=False, 
            persistent_workers=True, 
        )

    def test_dataloader(self):
        
        if self.use_tpu: 
            import torch_xla.core.xla_model as xm
            sampler = DDPSequnetialSampler(
                batch_size=self.chunk, 
                num_replicas=xm.xrt_world_size(), 
                rank=xm.get_ordinal(), 
                N_total=len(self.test_dset), 
                tpu=True
            )
        else:
            sampler = DDPSequnetialSampler(
                batch_size=self.chunk, 
                num_replicas=None, 
                rank=None, 
                N_total=len(self.test_dset), 
                tpu=False
            )
        
        return DataLoader(
            dataset=self.test_dset, 
            batch_size=self.chunk, 
            sampler=sampler,
            num_workers=self.num_workers, 
            pin_memory=True, 
            shuffle=False,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        if self.use_tpu: 
            import torch_xla.core.xla_model as xm
            sampler = DDPSequnetialSampler(
                batch_size=self.chunk, 
                num_replicas=xm.xrt_world_size(), 
                rank=xm.get_ordinal(), 
                N_total=len(self.predict_dset), 
                tpu=True
            )
        else:
            sampler = DDPSequnetialSampler(
                batch_size=self.chunk, 
                num_replicas=None, 
                rank=None, 
                N_total=len(self.predict_dset), 
                tpu=False
            )
        
        return DataLoader(
            dataset=self.predict_dset, 
            batch_size=self.chunk, 
            sampler=sampler,
            num_workers=self.num_workers, 
            pin_memory=True, 
            shuffle=False,
            persistent_workers=True,
        )