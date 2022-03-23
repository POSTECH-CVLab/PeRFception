import os

import numpy as np
import torch
import pytorch_lightning as pl

from model.interface import LitModel

import utils.store_image as store_image
import utils.ray as ray
import model.plenoxel_torch.sparse_grid as sparse_grid
import model.plenoxel_torch.utils as utils
import model.plenoxel_torch.dataclass as dataclass
import torch.nn as nn

from typing import *

from model.plenoxel_torch.__global__ import BASIS_TYPE_SH
import gin

class ResampleCallBack(pl.Callback):

    def __init__(
        self, 
    ):
        self.upsample_step = gin.query_parameter("LitPlenoxel.upsample_step")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if trainer.global_step > 0 and trainer.global_step in self.upsample_step \
            and pl_module.reso_idx + 1 < len(pl_module.reso_list):
            if pl_module.tv_early_only:
                pl_module.lambda_tv = 0.
                pl_module.lambda_tv_sh = 0.
            elif pl_module.tv_decay != 1.0:
                pl_module.lambda_tv *= pl_module.tv_decay
                pl_module.lambda_tv_sh *= pl_module.tv_decay

            pl_module.reso_idx += 1
            reso = pl_module.reso_list[pl_module.reso_idx]
            pl_module.model.resample(
                reso=reso,
                sigma_thresh=pl_module.density_thresh,
                weight_thresh=pl_module.weight_thresh / reso[2],
                dilate=2,
                cameras=pl_module.generate_camera_list() if \
                    pl_module.thresh_type == 'weight' else None,
                max_elements=pl_module.max_grid_elements,
            )

            if pl_module.model.use_background and pl_module.reso_idx <= 1:
                pl_module.model.sparsify_background(
                    pl_module.background_density_thresh)

            if pl_module.upsample_density_add:
                pl_module.model.density_data.data[:] += pl_module.upsample_density_add


@gin.configurable()
class LitPlenoxel(LitModel):

    # The external dataset will be called.
    def __init__(
        self, 
        reso: List[List[int]] = [[256, 256, 256], [512, 512, 512]],
        upsample_step: List[int] = [38400, 76800], 
        ndc_coeffs: List[int] = [-1., -1.],
        init_iters: int = 0,
        upsample_density_add: float = 0.0, 
        basis_type: str = "sh",
        sh_dim: int = 9,
        mlp_posenc_size: int = 4, 
        mlp_width: int = 32, 
        background_nlayers: int = 0,
        background_reso: int = 512,
        # Sigma Optim
        sigma_optim: str = "rmsprop", 
        lr_sigma: float = 3e1,
        lr_sigma_final: float = 5e-2,
        lr_sigma_decay_steps: int = 250000,
        lr_sigma_delay_steps: int = 15000,
        lr_sigma_delay_mult: float = 1e-2, 
        # SH Optim
        sh_optim: str = "rmsprop",
        lr_sh: float = 1e-2,
        lr_sh_final: float = 5e-6,
        lr_sh_decay_steps: int = 250000, 
        lr_sh_delay_steps: int = 0,
        lr_sh_delay_mult: float = 1e-2,
        lr_fg_begin_step: int = 0,
        # BG Simga Optim
        bg_optim: str = "rmsprop", 
        lr_sigma_bg: float = 3e0,
        lr_sigma_bg_final: float = 3e-3,
        lr_sigma_bg_decay_steps: int = 250000,
        lr_sigma_bg_delay_steps: int = 0, 
        lr_sigma_bg_delay_mult: float = 1e-2, 
        # BG Colors Optim
        lr_color_bg: float = 1e-1,
        lr_color_bg_final: float = 5e-6,
        lr_color_bg_decay_steps: int = 250000,
        lr_color_bg_delay_steps: int = 0, 
        lr_color_bg_delay_mult: float = 1e-2,
        # Basis Optim
        basis_optim: str = "rmsprop", 
        lr_basis: float = 1e-6,
        lr_basis_final: float = 1e-6,
        lr_basis_decay_steps: int = 250000,
        lr_basis_delay_steps: int = 0,
        lr_basis_begin_step: int = 0, 
        lr_basis_delay_mult: float = 1e-2,
        # RMSProp Option
        rms_beta: float = 0.95,
        # Init Option
        init_sigma: float = 0.1,
        init_sigma_bg: float = 0.1, 
        thresh_type: str = "weight", 
        weight_thresh: float = 0.0005 * 512,
        density_thresh: float = 5.0,
        background_density_thresh: float = 1.0 + 1e-9,
        max_grid_elements: int = 44_000_000,
        tune_mode: bool = False,
        tune_nosave: bool = False, 
        # Losses
        lambda_tv: float = 1e-5,
        tv_sparsity: float = 0.01,
        tv_logalpha: bool = False,
        lambda_tv_sh: float = 1e-3, 
        tv_sh_sparsity: float = 0.01, 
        lambda_tv_lumisphere: float = 0.0,
        tv_lumisphere_sparsity: float = 0.01,
        tv_lumisphere_dir_factor: float = 0.0, 
        tv_decay: float = 1.0,
        lambda_l2_sh: float = 0.0,
        tv_early_only: int = 1, 
        tv_contiguous: int = 1,
        # Other Lambdas
        lambda_sparsity: float = 0.0,
        lambda_beta: float = 0.0,
        lambda_tv_background_sigma: float = 1e-2,
        lmabda_tv_background_sparsity: float = 0.01, 
        lambda_tv_background_color: float = 1e-2,
        # WD
        weight_decay_sigma: float = 1.0,
        weight_decay_sh: float = 1.0,
        lr_decay: bool = True,
        n_train: Optional[int] = None,
        nosphereinit: bool = False,
        # Render Options
        step_size: float = 0.5,
        sigma_thresh: float = 1e-8,
        stop_thresh: float = 1e-7, 
        background_brightness: float = 1.0, 
        renderer_backend: str = "cuvol",
        random_sigma_std: float = 0.0, 
        random_sigma_std_background: float = 0.0, 
        near_clip: float = 0.00,
        use_spheric_clip: bool = False,
        enable_random: bool = False,
        last_sample_opaque: bool = False
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)
        
        super(LitPlenoxel, self).__init__()
        assert basis_type in ["sh", "3d_texture", "mlp"]
        assert sigma_optim in ["sgd", "rmsprop"]
        assert sh_optim in ["sgd", "rmsprop"]
        assert bg_optim in ["sgd", "rmsprop"]
        assert basis_optim in ["sgd", "rmsprop"]
        assert thresh_type in ["weight", "sigma"]
        assert renderer_backend in ["cuvol", "svox1", "nvol"]

        self.automatic_optimization = False
        self.reso_idx = 0
        self.reso_list = reso
        self.model = sparse_grid.SparseGrid(
            reso=self.reso_list[self.reso_idx],
            center=self.scene_center,
            radius=self.scene_radius,
            use_sphere_bound=self.use_sphere_bound and not nosphereinit,
            basis_dim=sh_dim,
            use_z_order=True,
            basis_type=eval("BASIS_TYPE_" + basis_type.upper()),
            mlp_posenc_size=mlp_posenc_size,
            mlp_width=mlp_width,
            background_nlayers=background_nlayers,
            background_reso=background_reso,
            device=self.device,
        )
        self.lr_sigma_func = self.get_expon_lr_func(
            lr_sigma,
            lr_sigma_final,
            lr_sigma_delay_steps,
            lr_sigma_delay_mult,
            lr_sigma_decay_steps,
        )
        self.lr_sh_func = self.get_expon_lr_func(
            lr_sh,
            lr_sh_final,
            lr_sh_delay_steps,
            lr_sh_delay_mult,
            lr_sh_decay_steps,
        )
        self.lr_sigma_bg_func = self.get_expon_lr_func(
            lr_sigma_bg,
            lr_sigma_bg_final,
            lr_sigma_bg_delay_steps,
            lr_sigma_bg_delay_mult,
            lr_sigma_bg_decay_steps,
        )
        self.lr_color_bg_func = self.get_expon_lr_func(
            lr_color_bg,
            lr_color_bg_final,
            lr_color_bg_delay_steps,
            lr_color_bg_delay_mult,
            lr_color_bg_decay_steps,
        )
        self.model.sh_data.data[:] = 0.0
        self.model.density_data.data[:] = 0. if lr_fg_begin_step > 0 else init_sigma
        if self.model.use_background:
            self.model.background_data.data[..., -1] = init_sigma_bg

    def generate_camera_list(self):
        extrinsics = self.extrinsics
        intrinsics = self.intrinsics
        ret = [
            dataclass.Camera(
                torch.from_numpy(extrinsics[i]).to(
                    dtype=torch.float32, device=self.device
                ),
                intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2],
                intrinsics[1, 2], self.w, self.h, self.dataset.ndc_coeffs
            )
            for i in self.i_train
        ]
        return ret

    def get_expon_lr_func(
        self, lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps
    ):
        def helper(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                # Disable this parameter
                return 0.0
            if lr_delay_steps > 0:
                # A kind of reverse cosine decay.
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp

        return helper

    def create_model(self):
        pass

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        gstep = self.trainer.global_step

        if self.lr_fg_begin_step > 0 and gstep == self.lr_fg_begin_step:
            self.model.density_data.data[:] = self.init_sigma

        lr_sigma = self.lr_sigma_func(gstep)
        lr_sh = self.lr_sh_func(gstep)
        lr_sigma_bg = self.lr_sigma_bg_func(gstep - self.lr_basis_begin_step)
        lr_color_bg = self.lr_color_bg_func(gstep - self.lr_basis_begin_step)

        rays, target = batch["ray"].to(torch.float32), batch["target"].to(
            torch.float32)
        
        if self.ndc_coeffs[0] != -1 or self.ndc_coeffs[1] != -1:
            rays = torch.stack(ray.convert_to_ndc(rays[:, 0], rays[:, 1], self.ndc_coeffs), dim=1)
        
        rays = dataclass.Rays(rays[:, 0].contiguous(), rays[:, 1].contiguous())
        
        rgb = self.model.volume_render_fused(
            rays,
            target,
            beta_loss=self.lambda_beta,
            sparsity_loss=self.lambda_sparsity,
            randomize=self.enable_random,
        )

        img_loss = utils.img2mse(rgb, target)
        psnr = utils.mse2psnr(img_loss)
        if gstep % 100 == 0:
            self.log("lr_sigma", lr_sigma, on_step=True)
            self.log("lr_sh", lr_sh, on_step=True)
            self.log("lr_sigma_bg", lr_sigma_bg, on_step=True)
            self.log("lr_color_bg", lr_color_bg, on_step=True)
            self.log("train_psnr", psnr, on_step=True, prog_bar=True, logger=True)

        if self.lambda_tv > 0.0:
            self.model.inplace_tv_grad(
                self.model.density_data.grad,
                scaling=self.lambda_tv,
                sparse_frac=self.tv_sparsity,
                logalpha=self.tv_logalpha,
                ndc_coeffs=self.dataset.ndc_coeffs,
                contiguous=self.tv_contiguous,
            )

        if self.lambda_tv_sh > 0.0:
            self.model.inplace_tv_color_grad(
                self.model.sh_data.grad,
                scaling=self.lambda_tv_sh,
                sparse_frac=self.tv_sh_sparsity,
                ndc_coeffs=self.dataset.ndc_coeffs,
                contiguous=self.tv_contiguous,
            )

        if self.lambda_tv_lumisphere > 0.0:
            self.model.inplace_tv_lumisphere_grad(
                self.model.sh_data.grad,
                scaling=self.lambda_tv_lumisphere,
                dir_factor=self.tv_lumisphere_dir_factor,
                sparse_frac=self.tv_lumisphere_sparsity,
                ndc_coeffs=self.dataset.ndc_coeffs,
            )

        if self.lambda_l2_sh > 0.0:
            self.model.inplace_l2_color_grad(
                self.model.sh_data.grad, scaling=self.lambda_l2_sh
            )

        if self.model.use_background and (
                self.lambda_tv_background_sigma > 0.0
                or self.lambda_tv_background_color > 0.0):

            self.model.inplace_tv_background_grad(
                self.model.background_data.grad,
                scaling=self.lambda_tv_background_color,
                scaling_density=self.lambda_tv_background_sigma,
                sparse_frac=self.tv_background_sparsity,
                contiguous=self.tv_contiguous,
            )

        if gstep >= self.lr_fg_begin_step:
            self.model.optim_density_step(
                lr_sigma, beta=self.rms_beta, optim=self.sigma_optim
            )
            self.model.optim_sh_step(
                lr_sh, beta=self.rms_beta, optim=self.sh_optim
            )

        if self.model.use_background:
            self.model.optim_background_step(
                lr_sigma_bg, lr_color_bg, beta=self.rms_beta, optim=self.bg_optim
            )

        if self.weight_decay_sh < 1.0 and gstep % 20 == 0:
            self.model.sh_data.data *= self.weight_decay_sigma
        if self.weight_decay_sigma < 1.0 and gstep % 20 == 0:
            self.model.density_data.data *= self.weight_decay_sh

    def render_rays(self, batch, batch_idx, cpu=False):
        ret = {}
        rays = batch["ray"].to(torch.float32)
        if "target" in batch.keys():
            target = batch["target"].to(torch.float32)
        else:
            target = torch.zeros(
                (len(batch["ray"]), 3), dtype=torch.float32, device=self.device
            ) + 0.5

        if self.ndc_coeffs[0] != -1 or self.ndc_coeffs[1] != -1:
            rays = torch.stack(ray.convert_to_ndc(rays[:, 0], rays[:, 1], self.ndc_coeffs), dim=1)
        rays = dataclass.Rays(rays[:, 0].contiguous(), rays[:, 1].contiguous())
        rgb = self.model.volume_render_fused(
            rays,
            target,
            beta_loss=self.lambda_beta,
            sparsity_loss=self.lambda_sparsity,
            randomize=False,
        )
        depth = self.model.volume_render_depth(rays, None)
        if cpu:
            rgb = rgb.detach().cpu()
            depth = depth.detach().cpu()
            target = target.detach().cpu()

        ret["rgb"] = rgb
        ret["depth"] = depth[:, None]
        if "target" in batch.keys():
            ret["target"] = target
        return ret

    def validation_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx, cpu=True)

    def predict_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx, cpu=True)

    def test_epoch_end(self, outputs):
        rgbs, target, depths = self.gather_results(outputs, self.test_dummy)
        del outputs
        if self.trainer.is_global_zero:
            np_rgbs = rgbs.view(-1, self.h, self.w, 3).numpy()
            np_target = target.view(-1, self.h, self.w, 3).numpy()
            np_depths = depths.view(-1, self.h, self.w).numpy()
            psnr = self.psnr(np_rgbs, np_target, self.i_train, self.i_val,
                                self.i_test)
            ssim = self.ssim(np_rgbs, np_target, self.i_train, self.i_val,
                                self.i_test)
            lpips = self.lpips(np_rgbs, np_target, self.i_train,
                                    self.i_val, self.i_test)
            image_dir = os.path.join(self.logdir, "render_model")
            os.makedirs(image_dir, exist_ok=True)
            store_image.store_image(image_dir, np_rgbs, np_depths)

            self.write_stats(os.path.join(self.logdir, "results.txt"), psnr,
                                ssim, lpips)

            self.write_stats(
                os.path.join(self.logdir, "results.json"), psnr, ssim, lpips
            )
            
    def on_predict_epoch_end(self, outputs):
        # In the prediction step, be sure to use outputs[0]
        # instead of outputs.
        rgbs, _, depths = self.gather_results(outputs[0], self.pred_dummy)
        del outputs

        if self.trainer.is_global_zero:
            np_rgbs = rgbs.view(-1, self.h, self.w, 3).detach().cpu().numpy()
            np_depths = depths.view(-1, self.h, self.w).detach().cpu().numpy()
            del rgbs, depths
            image_dir = os.path.join(self.logdir, "render_video")
            os.makedirs(image_dir, exist_ok=True)
            store_image.store_image(image_dir, np_rgbs, np_depths)
            store_image.store_video(image_dir, np_rgbs, np_depths)

    def validation_epoch_end(self, outputs):
        rgbs, target, _ = self.gather_results(outputs, self.val_dummy)
        rgbs, target = rgbs.reshape(-1, self.img_size * 3), target.reshape(
            -1, self.img_size * 3)
        mse = torch.mean((target - rgbs)**2, dim=1)
        psnr = -10.0 * torch.log(mse) / np.log(10)
        psnr_mean = psnr.mean()
        self.log("val_psnr", psnr_mean, on_epoch=True, sync_dist=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["reso_idx"] = self.reso_idx
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint) -> None:
        state_dict = checkpoint["state_dict"]

        self.reso_idx = checkpoint["reso_idx"]

        del self.model.basis_data
        del self.model.background_data
        del self.model.density_data
        del self.model.sh_data
        del self.model.links

        self.model.register_parameter(
            "basis_data",
            nn.Parameter(
                torch.zeros_like(
                    state_dict["model.basis_data"], dtype=torch.float32
                )
            )
        )
        self.model.register_parameter(
            "background_data",
            nn.Parameter(
                torch.zeros_like(
                    state_dict["model.background_data"], dtype=torch.float32
                )
            )
        )
        self.model.register_parameter(
            "density_data",
            nn.Parameter(
                torch.zeros_like(
                    state_dict["model.density_data"], dtype=torch.float32
                )
            )
        )
        self.model.register_parameter(
            "sh_data",
            nn.Parameter(
                torch.zeros_like(
                    state_dict["model.sh_data"], dtype=torch.float32
                )
            )
        )
        self.model.register_buffer(
            "links",
            torch.zeros_like(state_dict["model.links"], dtype=torch.int32)
        )
        if self.model.use_background:
            del self.model.background_links
            self.model.register_buffer(
                "background_links",
                torch.zeros_like(
                    state_dict["model.background_links"], dtype=torch.int32
                )
            )

        return super().on_load_checkpoint(checkpoint)


# class LitPlenoxelBlender(LitPlenoxel):
#     def __init__(self, args):
#         self.dataset = LitBlender(args)
#         self.scene_center = [0.0, 0.0, 0.0]
#         self.scene_radius = [1.0, 1.0, 1.0]
#         self.use_sphere_bound = True
#         super(LitPlenoxelBlender, self).__init__(args)


# class LitPlenoxelLLFF(LitPlenoxel):
#     def __init__(self, args):
#         self.dataset = LitLLFF(args)
#         K = self.dataset.intrinsics
#         radx = 1 + 2 * 250 / self.dataset.w
#         rady = 1 + 2 * 250 / self.dataset.h
#         radz = 1.0
#         self.scene_center = [0.0, 0.0, 0.0]
#         self.scene_radius = [radx, rady, radz]
#         self.use_sphere_bound = False
#         super(LitPlenoxelLLFF, self).__init__(args)


# class LitPlenoxelTnT(LitPlenoxel):
#     def __init__(self, args):
#         self.dataset = LitTnT(args)
#         self.scene_center = [0.0, 0.0, 0.0]
#         self.scene_radius = [1.0, 1.0, 1.0]
#         self.use_sphere_bound = True
#         super(LitPlenoxelTnT, self).__init__(args)

# class LitPlenoxelCo3D(LitPlenoxel):

#     def __init__(self, args):
#         self.dataset = LitCo3D(args)
#         self.scene_center = [0.0, 0.0, 0.0]
#         self.scene_radius = [1.0, 1.0, 1.0]
#         self.use_sphere_bound = True
#         super(LitPlenoxelCo3D, self).__init__(args)
