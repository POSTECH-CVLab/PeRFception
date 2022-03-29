from dataloader.data_util.llff import load_llff_data
from dataloader.data_util.blender import load_blender_data
from dataloader.data_util.tnt import load_tnt_data
from dataloader.data_util.nsvf import load_nsvf_data
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
            self.i_split,
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
        
        self.i_train, self.i_val, self.i_test, self.i_all = self.i_split

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
            self.i_split,
            self.render_poses
        ) = \
        load_blender_data(
            datadir=datadir, 
            scene_name=scene_name,
            test_skip=test_skip, 
            cam_scale_factor=cam_scale_factor,
            white_bkgd=white_bkgd,
        )
        
        self.i_train, self.i_val, self.i_test, self.i_all = self.i_split

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
        batch_size: int, 
        accelerator: bool,
        num_gpus: int,
        num_tpus: int,
        # TnT specific
        cam_scale_factor: float = 1.0,
    ):
        (
            self.images, 
            self.intrinsics, 
            self.extrinsics, 
            self.image_sizes, 
            self.near, 
            self.far,
            self.ndc_coeffs,
            self.i_split,
            self.render_poses
        ) = \
        load_tnt_data(
            datadir=datadir,
            cam_scale_factor=cam_scale_factor
        )
        self.i_train, self.i_val, self.i_test, self.i_all = self.i_split

        super(LitDataTnT, self).__init__(
            datadir=datadir,
            batch_size=batch_size,
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
        test_skip: int = 8, 
        cam_scale_factor: float = 0.95,
        white_bkgd: bool = True,
        data_bbox_scale = 1.1,
    ):

        (
            self.images, 
            self.intrinsics, 
            self.extrinsics, 
            self.image_sizes, 
            self.near, 
            self.far,
            self.ndc_coeffs,
            self.i_split,
            self.render_poses
        ) = \
        load_nsvf_data(
            datadir=datadir, 
            scene_name=scene_name,
            test_skip=test_skip, 
            cam_scale_factor=cam_scale_factor,
            white_bkgd=white_bkgd,
            data_bbox_scale=data_bbox_scale,
        )
        
        self.i_train, self.i_val, self.i_test, self.i_all = self.i_split

        super(LitDataNSVF, self).__init__(
            datadir=datadir,
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_tpus=num_tpus,
        )