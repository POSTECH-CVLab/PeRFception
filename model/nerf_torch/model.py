import os

import numpy as np

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.interface import LitModel

import model.nerf_torch.utils as utils
import model.nerf_torch.embedder as embedder
import utils.store_util as store_util

import gin
from typing import *

class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
    ):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips

        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] + [
            nn.Linear(W, W) if i not in
            self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)
        ])

        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W // 2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)

        return outputs


@gin.configurable()
class LitNeRF(LitModel):

    # The external dataset will be called.
    def __init__(
        self,
        netchunk: int = 1024*64,
        netdepth: int = 8,
        netwidth: int = 256, 
        netdepth_fine: int = 8,
        netwidth_fine: int = 256,
        # Render Option
        num_coarse_samples: int = 64, 
        num_fine_samples: int = 0, 
        perturb: float = 1.0, 
        i_embed: int = 0, 
        multires: int = 10, 
        multires_views: int = 4, 
        raw_noise_std: float = 0.0, 
        lr_init: float = 2.0e-4, 
        lr_final: float = 2.5e-6,
        lr_delay_steps: int = 0, 
        lr_delay_mult: float = 0.1,
        lindisp: bool = False,
        white_bkgd: bool = False
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)
        
        super(LitNeRF, self).__init__()

    def setup(self, stage: Optional[str] = None) -> None:
        
        if stage in ["fit", "test", None]:
            near = self.trainer.datamodule.near
            far = self.trainer.datamodule.far
            ndc_coeffs = self.trainer.datamodule.ndc_coeffs

            self.forward_fun = functools.partial(
                utils.render, 
                ndc_coeffs=ndc_coeffs
            )
            embed_fn, input_ch = embedder.get_embedder(self.multires, self.i_embed)

            input_ch_views = 0
            embed_viewdirs_fn = None
            embed_viewdirs_fn, input_ch_views = embedder.get_embedder(
                self.multires_views, self.i_embed
            )
            output_ch = 5 if self.num_fine_samples > 0 else 4
            skips = [4]
            self.model = NeRF(
                D=self.netdepth,
                W=self.netwidth,
                input_ch=input_ch,
                output_ch=output_ch,
                skips=skips,
                input_ch_views=input_ch_views,
            )

            if self.num_fine_samples > 0:
                self.model_fine = NeRF(
                    D=self.netdepth_fine,
                    W=self.netwidth_fine,
                    input_ch=input_ch,
                    output_ch=output_ch,
                    skips=skips,
                    input_ch_views=input_ch_views,
                )

            network_query_fn = lambda inputs, viewdirs, network_fn: utils.run_network(
                inputs,
                viewdirs,
                network_fn,
                embed_fn=embed_fn,
                embeddirs_fn=embed_viewdirs_fn,
                netchunk=self.netchunk,
            )

            render_kwargs_train = {
                "network_query_fn": network_query_fn,
                "perturb": self.perturb,
                "num_fine_samples": self.num_fine_samples,
                "network_fine": self.model_fine,
                "num_coarse_samples": self.num_coarse_samples,
                "network_fn": self.model,
                "white_bkgd": self.white_bkgd,
                "raw_noise_std": self.raw_noise_std,
                "near": near,
                "far": far
            }

        render_kwargs_test = {
            k: render_kwargs_train[k]
            for k in render_kwargs_train
        }
        render_kwargs_test['perturb'] = False
        render_kwargs_test['raw_noise_std'] = 0.

        self.start = 0
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test


    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
        )

    def forward_train(self, batch_rays):
        return self.forward_fun(
            rays=batch_rays,
            **self.render_kwargs_train
        )

    def forward_eval(self, batch_rays):
        return self.forward_fun(
            rays=batch_rays, 
            **self.render_kwargs_test
        )

    def training_step(self, batch, batch_idx):
        batch_rays = batch["ray"]
        target = batch["target"]
        rgb, disp, acc, depth, extras = self.forward_train(batch_rays)
        img_loss = utils.img2mse(rgb, target)
        loss1 = img_loss
        loss = loss1
        psnr = utils.mse2psnr(img_loss)

        if "rgb0" in extras:
            loss0 = utils.img2mse(extras["rgb0"], target)
            psnr0 = utils.mse2psnr(loss0)
            loss = loss + loss0
        
        self.log("train/psnr1", psnr, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True)

        return loss

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx,
        optimizer_closure, on_tpu, using_native_amp, using_lbfgs
    ):
        step = self.trainer.global_step
        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (
                1 - self.lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(
                    step / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.

        max_steps = gin.query_parameter("run.max_steps")
        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(
            np.log(self.lr_init) * (1 - t) +
            np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        optimizer.step(closure=optimizer_closure)

    def render_rays(self, batch, batch_idx):
        ret = {}
        batch_rays = batch["ray"]
        rgb, disparity, acc, _, extras = self.forward_eval(batch_rays)
        if "target" in batch:
            target = batch["target"]

        ret["rgb"] = rgb
        if "target" in batch:
            ret["target"] = target
        return ret

    def validation_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

    def test_epoch_end(self, outputs):
        dmodule = self.trainer.datamodule
        all_image_sizes = dmodule.all_image_sizes if \
            not dmodule.eval_test_only else dmodule.test_image_sizes
        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)
        psnr = self.psnr(
            rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        )
        ssim = self.ssim(
            rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        )
        lpips = self.lpips(
            rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        )
        
        self.log("test/psnr", psnr["test"], on_epoch=True, rank_zero_only=True)
        self.log("test/ssim", ssim["test"], on_epoch=True, rank_zero_only=True)
        self.log("test/lpips", lpips["test"], on_epoch=True, rank_zero_only=True)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(self.logdir, "render_model")
            os.makedirs(image_dir, exist_ok=True)
            store_util.store_image(image_dir, rgbs)

            self.write_stats(
                os.path.join(self.logdir, "results.json"), psnr, ssim, lpips
            )
            
            
        return psnr, ssim, lpips

    def on_predict_epoch_end(self, outputs):
        # In the prediction step, be sure to use outputs[0]
        # instead of outputs. 
        rgbs = self.alter_gather_cat(outputs, "rgb")
        targets = self.alter_gather_cat(outputs, "target")

        if self.trainer.is_global_zero: 
            rgbs = rgbs.view(-1, self.h, self.w, 3).detach().cpu().numpy()
            depths = depths.view(-1, self.h, self.w).detach().cpu().numpy() 
            image_dir = os.path.join(self.logdir, "render_video")
            os.makedirs(image_dir, exist_ok=True)
            store_util.store_image(image_dir, rgbs, depths)
            store_util.store_video(image_dir, rgbs, depths)

    def validation_epoch_end(self, outputs):
        val_image_sizes = self.trainer.datamodule.val_image_sizes
        rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
        psnr_mean = self.psnr_each(rgbs, targets).mean()
        ssim_mean = self.ssim_each(rgbs, targets).mean()
        lpips_mean = self.lpips_each(rgbs, targets).mean()
        self.log("val/psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/ssim", ssim_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/lpips", lpips_mean.item(), on_epoch=True, sync_dist=True)