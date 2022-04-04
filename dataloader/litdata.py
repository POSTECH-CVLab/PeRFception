from dataloader.data_util.llff import load_llff_data
from dataloader.data_util.blender import load_blender_data
from dataloader.data_util.tnt import load_tnt_data
from dataloader.data_util.nsvf import load_nsvf_data
from dataloader.data_util.co3d import load_co3d_data
from dataloader.interface import LitData
import gin

@gin.configurable()
class LitDataLLFF(LitData):

    def __init__(
        self,
        datadir: str,
        scene_name: str,
        accelerator: bool,
        num_gpus: int,
        num_tpus: int,
        # LLFF specific arguments
        factor: int = 4, 
        llffhold: int = 8, 
        spherify: bool = False,
        path_zflat: bool = False,
    ):
        ndc_coord = gin.query_parameter("LitData.ndc_coord")
        (
            self.images, 
            self.intrinsics, 
            self.extrinsics, 
            self.image_sizes, 
            self.near, 
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses
        ) = \
        load_llff_data(
            datadir=datadir, 
            scene_name=scene_name,
            factor=factor, 
            ndc_coord=ndc_coord,
            recenter=True, 
            bd_factor=0.75,
            spherify=spherify, 
            llffhold=llffhold,
            path_zflat=path_zflat,
        )

        super(LitDataLLFF, self).__init__(
            datadir=datadir,
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_tpus=num_tpus,
        )


@gin.configurable()
class LitDataBlender(LitData):
    
    def __init__(self,        
        datadir: str,
        scene_name: str,
        accelerator: bool,
        num_gpus: int,
        num_tpus: int,
        # Blender specific
        test_skip: int = 8, 
        cam_scale_factor: float = 1.0,
        white_bkgd: bool = True,
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
            self.render_poses
        ) = \
        load_blender_data(
            datadir=datadir, 
            scene_name=scene_name,
            test_skip=test_skip, 
            cam_scale_factor=cam_scale_factor,
            white_bkgd=white_bkgd,
        )

        super(LitDataBlender, self).__init__(
            datadir=datadir,
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_tpus=num_tpus,
        )


@gin.configurable()
class LitDataTnT(LitData):
    
    def __init__(
        self,         
        datadir: str,
        scene_name: str,
        accelerator: bool,
        num_gpus: int,
        num_tpus: int,
        # TnT specific
        cam_scale_factor: float = 0.95,
        val_skip: int = 8,
        test_skip: int = 8,
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
            self.render_poses
        ) = \
        load_tnt_data(
            datadir=datadir,
            scene_name=scene_name,
            cam_scale_factor=cam_scale_factor,
            val_skip=val_skip, 
            test_skip=test_skip
        )

        super(LitDataTnT, self).__init__(
            datadir=datadir,
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_tpus=num_tpus,
        )

@gin.configurable()
class LitDataNSVF(LitData):
    
    def __init__(self,        
        datadir: str,
        scene_name: str,
        accelerator: bool,
        num_gpus: int,
        num_tpus: int,
        # NSVF specific
        val_skip: int = 8,
        test_skip: int = 8, 
        cam_scale_factor: float = 0.95,
        data_bbox_scale: float = 1.1,
        white_bkgd: bool = True,
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
            self.render_poses
        ) = \
        load_nsvf_data(
            datadir=datadir, 
            scene_name=scene_name,
            val_skip=val_skip,
            test_skip=test_skip, 
            cam_scale_factor=cam_scale_factor,
            data_bbox_scale=data_bbox_scale,
            white_bkgd=white_bkgd,
        )
        
        super(LitDataNSVF, self).__init__(
            datadir=datadir,
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_tpus=num_tpus,
        )


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
        cam_scale_factor: float = 0.95,
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
            self.render_poses
        ) = \
        load_co3d_data(
            datadir=datadir, 
            scene_name=scene_name,
            max_image_dim=max_image_dim,
            cam_scale_factor=cam_scale_factor
        )

        super(LitDataCo3D, self).__init__(
            datadir=datadir,
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_tpus=num_tpus,
        )