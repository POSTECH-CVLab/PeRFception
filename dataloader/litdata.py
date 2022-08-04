from dataloader.data_util.co3d import load_co3d_data
from dataloader.interface import LitData
import gin

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
            self.label_info
        ) = \
        load_co3d_data(
            datadir=datadir, 
            scene_name=scene_name,
            max_image_dim=max_image_dim,
            cam_scale_factor=cam_scale_factor
        )

        self.render_scale = 300 / max(self.image_sizes[0][0], self.image_sizes[0][1])

        super(LitDataCo3D, self).__init__(
            datadir=datadir,
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_tpus=num_tpus,
        )