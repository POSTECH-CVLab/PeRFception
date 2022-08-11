import os

import gin
import numpy as np

from dataloader.data_util.co3d import load_co3d_data
from dataloader.data_util.scannet import load_scannet_data
from dataloader.interface import LitData


@gin.configurable()
class LitDataCo3D(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        accelerator: bool,
        num_gpus: int,
        num_tpus: int,
        # Co3D specific arguments
        max_image_dim: int = 800,
        cam_scale_factor: float = 1.50,
    ):
        (
            self.images,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
            self.label_info,
        ) = load_co3d_data(
            datadir=datadir,
            scene_name=scene_name,
            max_image_dim=max_image_dim,
            cam_scale_factor=cam_scale_factor,
        )

        self.render_scale = 300 / max(self.image_sizes[0][0], self.image_sizes[0][1])

        super(LitDataCo3D, self).__init__(
            datadir=datadir,
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_tpus=num_tpus,
        )


@gin.configurable()
class LitDataScannet(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        accelerator: bool,
        num_gpus: int,
        num_tpus: int,
        # scannet specific arguments
        frame_skip: int = 1,
        max_frame: int = 1500,
        max_image_dim: int = 800,
        cam_scale_factor: float = 1.50,
        use_depth: bool = True,
        use_scans: bool = True,
        blur_thresh: float = 10.0,
        pcd_name: str = "tsdf_pcd.pcd",
    ):
        super(LitDataScannet, self).__init__(
            datadir=datadir,
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_tpus=num_tpus,
        )

        (
            images,
            extrinsics,
            render_poses,
            (h, w),
            intrinsics,
            i_split,
            depths,
            trans_info,
        ) = load_scannet_data(
            os.path.join(datadir, scene_name),
            cam_scale_factor=cam_scale_factor,
            frame_skip=frame_skip,
            max_frame=max_frame,
            max_image_dim=max_image_dim,
            blur_thresh=blur_thresh,
            use_depth=use_depth,
            use_scans=use_scans,
            pcd_name=pcd_name,
        )
        i_train, i_val, i_test = i_split

        print(f"loaded scannet, image with size: {h} * {w}")
        self.scene_name = scene_name
        self.images = images
        self.intrinsics = intrinsics.reshape(-1, 4, 4).repeat(len(images), axis=0)
        self.extrinsics = extrinsics
        self.image_sizes = np.array([h, w]).reshape(1, 2).repeat(len(images), axis=0)
        self.near = 0.0
        self.far = 1.0
        self.ndc_coeffs = (-1.0, -1.0)
        self.i_train, self.i_val, self.i_test = i_train, i_val, i_test
        self.i_all = np.arange(len(images))
        self.render_poses = render_poses
        self.trans_info = trans_info
        self.use_sphere_bound = False
