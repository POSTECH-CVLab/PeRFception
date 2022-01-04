import os

import numpy as np
import torch
import pytorch_lightning as pl
from dataloader.tnt import LitTnT

from model.interface import LitModel
from dataloader.blender import LitBlender
from dataloader.llff import LitLLFF

import utils.metrics as metrics
import utils.store_image as store_image
import model.plenoxel_torch.sparse_grid as sparse_grid
import model.plenoxel_torch.utils as utils
import model.plenoxel_torch.dataclass as dataclass
import torch.nn as nn

from config import str2bool

from model.plenoxel_torch.__global__ import (
    BASIS_TYPE_3D_TEXTURE,
    BASIS_TYPE_MLP,
    BASIS_TYPE_SH,
)

class ResampleCallBack(pl.Callback):

    def __init__(self, args):
        self.args = args
        self.upsamp_every = args.upsamp_every

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > 0 and trainer.global_step % self.upsamp_every == 0 \
            and pl_module.reso_idx + 1 < len(pl_module.reso_list):
            if pl_module.args.tv_early_only:
                pl_module.args.lambda_tv = 0.
                pl_module.args.lambda_tv_sh = 0.
            elif pl_module.args.tv_decay != 1.0: 
                pl_module.args.lambda_tv *= pl_module.args.tv_decay
                pl_module.args.lambda_tv_sh *= pl_module.args.tv_decay
            
            pl_module.reso_idx += 1
            reso = pl_module.reso_list[pl_module.reso_idx]
            pl_module.model.resample(
                reso=reso,
                sigma_thresh=pl_module.args.density_thresh,
                weight_thresh=pl_module.args.weight_thresh / reso[2],
                dilate=2, 
                cameras=pl_module.generate_camera_list() if \
                    pl_module.args.thresh_type == 'weight' else None,
                max_elements=pl_module.args.max_grid_elements,
                GL=pl_module.GL
            )

            if pl_module.model.use_background and pl_module.reso_idx <= 1:
                pl_module.model.sparsify_background(pl_module.args.background_density_thresh)

            if pl_module.args.upsample_density_add:
                pl_module.model.density_data.data[:] += pl_module.args.upsample_density_add

class LitPlenoxel(LitModel):

    # The external dataset will be called.
    def __init__(self, args):
        super(LitPlenoxel, self).__init__(args)

        self.automatic_optimization = False
        self.reso_idx = 0
        self.reso_list = eval(args.reso)
        self.model = sparse_grid.SparseGrid(
            args=args,
            reso=self.reso_list[self.reso_idx],
            center=self.scene_center,
            radius=self.scene_radius,
            use_sphere_bound=self.use_sphere_bound and not args.nosphereinit,
            basis_dim=args.sh_dim,
            use_z_order=True,
            basis_reso=self.args.basis_reso,
            basis_type=eval("BASIS_TYPE_" + args.basis_type.upper()),
            mlp_posenc_size=args.mlp_posenc_size,
            mlp_width=args.mlp_width,
            background_nlayers=args.background_nlayers,
            background_reso=args.background_reso,
            device=self.device,
        )
        self.lr_sigma_func = self.get_expon_lr_func(
            args.lr_sigma,
            args.lr_sigma_final,
            args.lr_sigma_delay_steps,
            args.lr_sigma_delay_mult,
            args.lr_sigma_decay_steps,
        )
        self.lr_sh_func = self.get_expon_lr_func(
            args.lr_sh,
            args.lr_sh_final,
            args.lr_sh_delay_steps,
            args.lr_sh_delay_mult,
            args.lr_sh_decay_steps,
        )
        self.lr_sigma_bg_func = self.get_expon_lr_func(
            args.lr_sigma_bg,
            args.lr_sigma_bg_final,
            args.lr_sigma_bg_delay_steps,
            args.lr_sigma_bg_delay_mult,
            args.lr_sigma_bg_decay_steps,
        )
        self.lr_color_bg_func = self.get_expon_lr_func(
            args.lr_color_bg,
            args.lr_color_bg_final,
            args.lr_color_bg_delay_steps,
            args.lr_color_bg_delay_mult,
            args.lr_color_bg_decay_steps,
        )
        self.model.sh_data.data[:] = 0.0
        self.model.density_data.data[:] = 0. if args.lr_fg_begin_step > 0 else args.init_sigma
        if self.model.use_background:
            self.model.background_data.data[..., -1] = args.init_sigma_bg

    def get_expon_lr_func(
        self, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
    ):
        def helper(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                # Disable this parameter
                return 0.0
            if lr_delay_steps > 0:
                # A kind of reverse cosine decay.
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                )
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

    def generate_camera_list(self):
        extrinsics = self.extrinsics
        intrinsics = self.intrinsics
        ret = [
            dataclass.Camera(
                torch.from_numpy(extrinsics[i]).to(dtype=torch.float32, device=self.device), 
                intrinsics[0, 0], intrinsics[1, 1], 
                intrinsics[0, 2], intrinsics[1, 2], self.w, self.h,
                self.ndc_coeffs
            ) for i in self.i_train
        ]
        return ret

    def convert_to_ndc(self, origins, directions, ndc_coeffs, near: float = 1.0):
        """Convert a set of rays to NDC coordinates."""
        # Shift ray origins to near plane, not sure if needed
        # Projection
        if self.GL:
            t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
            origins = origins + t[Ellipsis, None] * directions

            dx, dy, dz = directions.unbind(-1)
            ox, oy, oz = origins.unbind(-1)
            o0 = - ndc_coeffs[0] * (ox / oz)
            o1 =-  ndc_coeffs[1] * (oy / oz)
            o2 = 1 + 2 * near / oz
            d0 = - ndc_coeffs[0] * (dx / dz - ox / oz)
            d1 = - ndc_coeffs[1] * (dy / dz - oy / oz)
            d2 = -2 * near / oz;
        else:
            t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
            origins = origins + t[Ellipsis, None] * directions

            dx, dy, dz = directions.unbind(-1)
            ox, oy, oz = origins.unbind(-1)
            o0 = ndc_coeffs[0] * (ox / oz)
            o1 = ndc_coeffs[1] * (oy / oz)
            o2 = 1 - 2 * near / oz
            d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
            d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
            d2 = 2 * near / oz;

        origins = torch.stack([o0, o1, o2], -1)
        directions = torch.stack([d0, d1, d2], -1)
        return origins, directions


    def training_step(self, batch, batch_idx):
        args = self.args
        gstep = self.trainer.global_step

        if args.lr_fg_begin_step > 0 and gstep == args.lr_fg_begin_step:
            self.model.density_data.data[:] = args.init_sigma

        lr_sigma = self.lr_sigma_func(gstep)
        lr_sh = self.lr_sh_func(gstep)
        lr_sigma_bg = self.lr_sigma_bg_func(gstep - args.lr_basis_begin_step)
        lr_color_bg = self.lr_color_bg_func(gstep - args.lr_basis_begin_step)

        rays, target = batch["ray"].to(torch.float32), batch["target"].to(torch.float32)

        if self.ndc_coeffs[0] != -1:
            rays_o, rays_d = self.convert_to_ndc(rays[:, 0], rays[:, 1], self.ndc_coeffs)
            rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
            rays = torch.stack([rays_o, rays_d], dim=1)

        rays = dataclass.Rays(rays[:, 0].contiguous(), rays[:, 1].contiguous())
        rgb = self.model.volume_render_fused(
            rays,
            target,
            beta_loss=self.args.lambda_beta,
            sparsity_loss=self.args.lambda_sparsity,
            randomize=self.args.enable_random,
        )

        img_loss = utils.img2mse(rgb, target)
        psnr = utils.mse2psnr(img_loss)
        if gstep % 100 == 0:
            self.log("lr_sigma", lr_sigma, on_step=True)
            self.log("lr_sh", lr_sh, on_step=True)
            self.log("lr_sigma_bg", lr_sigma_bg, on_step=True)
            self.log("lr_color_bg", lr_color_bg, on_step=True)
            self.log("train_psnr", psnr, on_step=True, prog_bar=True, logger=True)

        if args.lambda_tv > 0.0:
            self.model.inplace_tv_grad(
                self.model.density_data.grad,
                scaling=args.lambda_tv,
                sparse_frac=args.tv_sparsity,
                logalpha=args.tv_logalpha,
                ndc_coeffs=self.ndc_coeffs,
                contiguous=args.tv_contiguous,
            )

        if args.lambda_tv_sh > 0.0:
            self.model.inplace_tv_color_grad(
                self.model.sh_data.grad,
                scaling=args.lambda_tv_sh,
                sparse_frac=args.tv_sh_sparsity,
                ndc_coeffs=self.ndc_coeffs,
                contiguous=args.tv_contiguous,
            )

        if args.lambda_tv_lumisphere > 0.0:
            self.model.inplace_tv_lumisphere_grad(
                self.model.sh_data.grad,
                scaling=args.lambda_tv_lumisphere,
                dir_factor=args.tv_lumisphere_dir_factor,
                sparse_frac=args.tv_lumisphere_sparsity,
                ndc_coeffs=self.ndc_coeffs,
            )

        if args.lambda_l2_sh > 0.0:
            self.model.inplace_l2_color_grad(
                self.model.sh_data.grad, scaling=args.lambda_l2_sh
            )

        if self.model.use_background and (
            args.lambda_tv_background_sigma > 0.0
            or args.lambda_tv_background_color > 0.0
        ):
            self.model.inplace_tv_background_grad(
                self.model.background_data.grad,
                scaling=args.lambda_tv_background_color,
                scaling_density=args.lambda_tv_background_sigma,
                sparse_frac=args.tv_background_sparsity,
                contiguous=args.tv_contiguous,
            )

        if args.lambda_tv_basis > 0.0:
            tv_basis = self.model.tv_basis()
            loss_tv_basis = tv_basis * args.lambda_tv_basis
            loss_tv_basis.backward()

        if gstep >= args.lr_fg_begin_step:
            self.model.optim_density_step(
                lr_sigma, beta=args.rms_beta, optim=args.sigma_optim
            )
            self.model.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)

        if self.model.use_background:
            self.model.optim_background_step(
                lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim
            )

        if args.weight_decay_sh < 1.0 and gstep % 20 == 0:
            self.model.sh_data.data *= args.weight_decay_sigma
        if args.weight_decay_sigma < 1.0 and gstep % 20 == 0:
            self.model.density_data.data *= args.weight_decay_sh

    @torch.no_grad()
    def render_rays(self, batch, batch_idx):
        ret = {}
        rays = batch["ray"].to(torch.float32)
        if self.ndc_coeffs[0] != -1:
            rays_o, rays_d = self.convert_to_ndc(rays[:, 0], rays[:, 1], self.ndc_coeffs)
            rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
            rays = torch.stack([rays_o, rays_d], dim=1)
        
        if "target" in batch.keys():
            target = batch["target"].to(torch.float32)
        else:
            target = torch.zeros((len(batch["ray"]), 3), dtype=torch.float32, device=self.device) + 0.5
        rays = dataclass.Rays(rays[:, 0].contiguous(), rays[:, 1].contiguous())
        rgb = self.model.volume_render_fused(
            rays,
            target,
            beta_loss=self.args.lambda_beta,
            sparsity_loss=self.args.lambda_sparsity,
            randomize=False,
        )
        depth = self.model.volume_render_depth(rays, None)
        ret["rgb"] = rgb
        ret["depth"] = depth[:, None]
        if "target" in batch.keys():
            ret["target"] = batch["target"]
        return ret

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def test_epoch_end(self, outputs):
        rgbs, target, depths = self.gather_results(outputs, self.test_dummy)

        if self.trainer.is_global_zero:
            rgbs = rgbs.reshape(-1, self.h, self.w, 3).detach().cpu().numpy()
            target = target.reshape(-1, self.h, self.w, 3).detach().cpu().numpy()
            depths = depths.reshape(-1, self.h, self.w).detach().cpu().numpy()
            psnr = metrics.psnr(rgbs, target, self.i_train, self.i_val, self.i_test)
            ssim = metrics.ssim(rgbs, target, self.i_train, self.i_val, self.i_test)
            lpips = metrics.lpips_v(rgbs, target, self.i_train, self.i_val, self.i_test)
            image_dir = os.path.join(self.logdir, "render_model")
            os.makedirs(image_dir, exist_ok=True)
            store_image.store_image(image_dir, rgbs, depths)

            metrics.write_stats(
                os.path.join(self.logdir, "results.txt"), psnr, ssim, lpips
            )

    def on_predict_epoch_end(self, outputs):
        # In the prediction step, be sure to use outputs[0]
        # instead of outputs.
        rgbs, _, depths = self.gather_results(outputs[0], self.pred_dummy)
        if self.trainer.is_global_zero:
            rgbs = rgbs.reshape(-1, self.h, self.w, 3).detach().cpu().numpy()
            depths = depths.reshape(-1, self.h, self.w).detach().cpu().numpy()
            image_dir = os.path.join(self.logdir, "render_video")
            os.makedirs(image_dir, exist_ok=True)
            store_image.store_image(image_dir, rgbs, depths)
            store_image.store_video(image_dir, rgbs, depths)

    def validation_epoch_end(self, outputs):
        rgbs, target, _ = self.gather_results(outputs, self.val_dummy)
        rgbs, target = rgbs.reshape(-1, self.img_size * 3), target.reshape(
            -1, self.img_size * 3
        )
        mse = torch.mean((target - rgbs) ** 2, dim=1)
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
            nn.Parameter(torch.zeros_like(state_dict["model.basis_data"], dtype=torch.float32))
        )
        self.model.register_parameter(
            "background_data", 
            nn.Parameter(torch.zeros_like(state_dict["model.background_data"], dtype=torch.float32))
        )
        self.model.register_parameter(
            "density_data", 
            nn.Parameter(torch.zeros_like(state_dict["model.density_data"], dtype=torch.float32))
        )
        self.model.register_parameter(
            "sh_data", 
            nn.Parameter(torch.zeros_like(state_dict["model.sh_data"], dtype=torch.float32))
        )
        self.model.register_buffer(
            "links", 
            torch.zeros_like(state_dict["model.links"], dtype=torch.int32)
        )
        if self.model.use_background:
            del self.model.background_links
            self.model.register_buffer(
                "background_links", nn.Parameter(
                    torch.zeros_like(
                        state_dict["model.background_links"], dtype=torch.int32
                    )
                )
            )

        return super().on_load_checkpoint(checkpoint)

    @staticmethod
    def add_model_specific_args(parser):

        group = parser.add_argument_group("general")
        group.add_argument(
            "--reso",
            type=str,
            default="[[256, 256, 256], [512, 512, 512]]",
            help="""
            List of grid resolution (will be evaled as json);
            resamples to the next one every upsamp_every iters, then
            stays at the last one; 
            should be a list where each item is a list of 3 ints or an int'
            """,
        )
        group.add_argument(
            "--upsamp_every",
            type=int,
            default=3 * 12800,
            help="upsample the grid every x iters",
        )
        group.add_argument(
            "--init_iters",
            type=int,
            default=0,
            help="do not upsample for first x iters",
        )
        group.add_argument(
            "--upsample_density_add",
            type=float,
            default=0.0,
            help="add the remaining density by this amount when upsampling",
        )
        group.add_argument(
            "--basis_type",
            choices=["sh", "3d_texture", "mlp"],
            default="sh",
            help="Basis function type",
        )
        group.add_argument(
            "--basis_reso",
            type=int,
            default=32,
            help="basis grid resolution (only for learned texture)",
        )
        group.add_argument(
            "--sh_dim",
            type=int,
            default=9,
            help="SH/learned basis dimensions (at most 10)",
        )

        group.add_argument(
            "--mlp_posenc_size",
            type=int,
            default=4,
            help="Positional encoding size if using MLP basis; 0 to disable",
        )
        group.add_argument(
            "--mlp_width", type=int, default=32, help="MLP width if using MLP basis"
        )

        group.add_argument(
            "--background_nlayers",
            type=int,
            default=0,
            help="Number of background layers (0=disable BG model)",
        )
        group.add_argument(
            "--background_reso", type=int, default=512, help="Background resolution"
        )

        group = parser.add_argument_group("optimization")

        group.add_argument(
            "--sigma_optim",
            choices=["sgd", "rmsprop"],
            default="rmsprop",
            help="Density optimizer",
        )
        group.add_argument(
            "--lr_sigma", type=float, default=3e1, help="SGD/rmsprop lr for sigma"
        )
        group.add_argument("--lr_sigma_final", type=float, default=5e-2)
        group.add_argument("--lr_sigma_decay_steps", type=int, default=250000)
        group.add_argument(
            "--lr_sigma_delay_steps",
            type=int,
            default=15000,
            help="Reverse cosine steps (0 means disable)",
        )
        group.add_argument("--lr_sigma_delay_mult", type=float, default=1e-2)

        group.add_argument(
            "--sh_optim",
            choices=["sgd", "rmsprop"],
            default="rmsprop",
            help="SH optimizer",
        )
        group.add_argument(
            "--lr_sh", type=float, default=1e-2, help="SGD/rmsprop lr for SH"
        )
        group.add_argument("--lr_sh_final", type=float, default=5e-6)
        group.add_argument("--lr_sh_decay_steps", type=int, default=250000)
        group.add_argument(
            "--lr_sh_delay_steps",
            type=int,
            default=0,
            help="Reverse cosine steps (0 means disable)",
        )
        group.add_argument("--lr_sh_delay_mult", type=float, default=1e-2)
        group.add_argument(
            "--lr_fg_begin_step",
            type=int,
            default=0,
            help="Foreground begins training at given step number",
        )

        # BG LRs
        group.add_argument(
            "--bg_optim",
            choices=["sgd", "rmsprop"],
            default="rmsprop",
            help="Background optimizer",
        )
        group.add_argument(
            "--lr_sigma_bg",
            type=float,
            default=3e0,
            help="SGD/rmsprop lr for background",
        )
        group.add_argument(
            "--lr_sigma_bg_final",
            type=float,
            default=3e-3,
            help="SGD/rmsprop lr for background",
        )
        group.add_argument("--lr_sigma_bg_decay_steps", type=int, default=250000)
        group.add_argument(
            "--lr_sigma_bg_delay_steps",
            type=int,
            default=0,
            help="Reverse cosine steps (0 means disable)",
        )
        group.add_argument("--lr_sigma_bg_delay_mult", type=float, default=1e-2)

        group.add_argument(
            "--lr_color_bg",
            type=float,
            default=1e-1,
            help="SGD/rmsprop lr for background",
        )
        group.add_argument(
            "--lr_color_bg_final",
            type=float,
            default=5e-6,
            help="SGD/rmsprop lr for background",
        )
        group.add_argument("--lr_color_bg_decay_steps", type=int, default=250000)
        group.add_argument(
            "--lr_color_bg_delay_steps",
            type=int,
            default=0,
            help="Reverse cosine steps (0 means disable)",
        )
        group.add_argument("--lr_color_bg_delay_mult", type=float, default=1e-2)

        group.add_argument(
            "--basis_optim",
            choices=["sgd", "rmsprop"],
            default="rmsprop",
            help="Learned basis optimizer",
        )
        group.add_argument(
            "--lr_basis", type=float, default=1e-6, help="SGD/rmsprop lr for SH"
        )
        group.add_argument("--lr_basis_final", type=float, default=1e-6)
        group.add_argument("--lr_basis_decay_steps", type=int, default=250000)
        group.add_argument(
            "--lr_basis_delay_steps",
            type=int,
            default=0,
            help="Reverse cosine steps (0 means disable)",
        )
        group.add_argument("--lr_basis_begin_step", type=int, default=0)
        group.add_argument("--lr_basis_delay_mult", type=float, default=1e-2)

        group.add_argument(
            "--rms_beta",
            type=float,
            default=0.95,
            help="RMSProp exponential averaging factor",
        )

        group.add_argument(
            "--init_sigma", type=float, default=0.1, help="initialization sigma"
        )
        group.add_argument(
            "--init_sigma_bg",
            type=float,
            default=0.1,
            help="initialization sigma (for BG)",
        )

        group = parser.add_argument_group("misc experiments")
        group.add_argument(
            "--thresh_type",
            choices=["weight", "sigma"],
            default="weight",
            help="Upsample threshold type",
        )
        group.add_argument(
            "--weight_thresh",
            type=float,
            default=0.0005 * 512,
            #  default=0.025 * 512,
            help="Upsample weight threshold; will be divided by resulting z-resolution",
        )
        group.add_argument(
            "--density_thresh", type=float, default=5.0, help="Upsample sigma threshold"
        )
        group.add_argument(
            "--background_density_thresh",
            type=float,
            default=1.0 + 1e-9,
            help="Background sigma threshold for sparsification",
        )
        group.add_argument(
            "--max_grid_elements",
            type=int,
            default=44_000_000,
            help="Max items to store after upsampling "
            "(the number here is given for 22GB memory)",
        )

        group.add_argument(
            "--tune_mode",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="hypertuning mode (do not save, for speed)",
        )
        group.add_argument(
            "--tune_nosave",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="do not save any checkpoint even at the end",
        )

        group = parser.add_argument_group("losses")
        # Foreground TV
        group.add_argument("--lambda_tv", type=float, default=1e-5)
        group.add_argument("--tv_sparsity", type=float, default=0.01)
        group.add_argument(
            "--tv_logalpha",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Use log(1-exp(-delta * sigma)) as in neural volumes",
        )

        group.add_argument("--lambda_tv_sh", type=float, default=1e-3)
        group.add_argument("--tv_sh_sparsity", type=float, default=0.01)

        group.add_argument("--lambda_tv_lumisphere", type=float, default=0.0)
        group.add_argument("--tv_lumisphere_sparsity", type=float, default=0.01)
        group.add_argument("--tv_lumisphere_dir_factor", type=float, default=0.0)

        group.add_argument("--tv_decay", type=float, default=1.0)

        group.add_argument("--lambda_l2_sh", type=float, default=0.0)  # 1e-4)
        group.add_argument(
            "--tv_early_only",
            type=int,
            default=1,
            help="Turn off TV regularization after the first split/prune",
        )

        group.add_argument(
            "--tv_contiguous",
            type=int,
            default=1,
            help="Apply TV only on contiguous link chunks, which is faster",
        )
        # End Foreground TV

        group.add_argument(
            "--lambda_sparsity",
            type=float,
            default=0.0,
            help="""
            Weight for sparsity loss as in SNeRG/PlenOctrees
            (but applied on the ray)
            """,
        )
        group.add_argument(
            "--lambda_beta",
            type=float,
            default=0.0,
            help="Weight for beta distribution sparsity loss as in neural volumes",
        )

        # Background TV
        group.add_argument("--lambda_tv_background_sigma", type=float, default=1e-2)
        group.add_argument("--lambda_tv_background_color", type=float, default=1e-2)

        group.add_argument("--tv_background_sparsity", type=float, default=0.01)
        # End Background TV

        # Basis TV
        group.add_argument(
            "--lambda_tv_basis",
            type=float,
            default=0.0,
            help="Learned basis total variation loss",
        )
        # End Basis TV

        group.add_argument("--weight_decay_sigma", type=float, default=1.0)
        group.add_argument("--weight_decay_sh", type=float, default=1.0)
        group.add_argument(
            "--lr_decay", 
            type=str2bool,
            nargs="?",
            const=True,
            default=True
        )

        group.add_argument(
            "--n_train",
            type=int,
            default=None,
            help="Number of training images. Defaults to use all avaiable.",
        )

        group.add_argument(
            "--nosphereinit",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="do not start with sphere bounds (please do not use for 360)",
        )

        group = parser.add_argument_group("Render options")
        group.add_argument(
            "--step_size",
            type=float,
            default=0.5,
            help="Render step size (in voxel size units)",
        )
        group.add_argument(
            "--sigma_thresh",
            type=float,
            default=1e-8,
            help="Skips voxels with sigma < this",
        )
        group.add_argument(
            "--stop_thresh",
            type=float,
            default=1e-7,
            help="Ray march stopping threshold",
        )
        group.add_argument(
            "--background_brightness",
            type=float,
            default=1.0,
            help="Brightness of the infinite background",
        )
        group.add_argument(
            "--renderer_backend",
            "-B",
            choices=["cuvol", "svox1", "nvol"],
            default="cuvol",
            help="Renderer backend",
        )
        group.add_argument(
            "--random_sigma_std",
            type=float,
            default=0.0,
            help="Random Gaussian std to add to density values (only if enable_random)",
        )
        group.add_argument(
            "--random_sigma_std_background",
            type=float,
            default=0.0,
            help="Random Gaussian std to add to density values for BG (only if enable_random)",
        )
        group.add_argument(
            "--near_clip",
            type=float,
            default=0.00,
            help="Near clip distance (in world space distance units, only for FG)",
        )
        group.add_argument(
            "--use_spheric_clip",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Use spheric ray clipping instead of voxel grid AABB "
            "(only for FG; changes near_clip to mean 1-near_intersection_radius; "
            "far intersection is always at radius 1)",
        )
        group.add_argument(
            "--enable_random",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Random Gaussian std to add to density values",
        )
        group.add_argument(
            "--last_sample_opaque",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Last sample has +1e9 density (used for LLFF)",
        )

        return parser.parse_args()


class LitPlenoxelBlender(LitPlenoxel):
    def __init__(self, args):
        self.dataset = LitBlender(args)
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = [1.0, 1.0, 1.0]
        self.use_sphere_bound = True
        self.ndc_coeffs = (-1, -1)
        super(LitPlenoxelBlender, self).__init__(args)


class LitPlenoxelLLFF(LitPlenoxel):
    def __init__(self, args):
        self.dataset = LitLLFF(args)
        K = self.dataset.intrinsics
        radx = 1 + 2 * 250 / self.dataset.w
        rady = 1 + 2 * 250 / self.dataset.h
        radz = 1.0
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = [radx, rady, radz]
        self.use_sphere_bound = False
        self.ndc_coeffs = (2 * K[0][0] / self.dataset.w, 2 * K[1][1] / self.dataset.h)
        super(LitPlenoxelLLFF, self).__init__(args)


class LitPlenoxelTnT(LitPlenoxel):
    def __init__(self, args):
        self.dataset = LitTnT(args)
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = [1.0, 1.0, 1.0]
        self.use_sphere_bound = True
        self.ndc_coeffs = (-1, -1)
        super(LitPlenoxelTnT, self).__init__(args)
