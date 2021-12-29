import os

import numpy as np

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.interface import LitModel

import model.jaxnerf_torch.utils as utils
import model.jaxnerf_torch.embedder as embedder
import pytorch_lightning as pl
import utils.metrics as metrics
import utils.store_image as store_image
from dataloader.blender import LitBlender
from dataloader.llff import LitLLFF


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


class LitJaxNeRF(LitModel):

    # The external dataset will be called.
    def __init__(self, args):
        super(LitJaxNeRF, self).__init__(args)
        self.forward_fun = functools.partial(
            utils.render, H=self.h, W=self.w, K=self.intrinsics,
            use_pixel_centers=self.args.use_pixel_centers
        )

    @staticmethod
    def add_model_specific_args(parser):
        
        # ray option
        ray = parser.add_argument_group("rays") 
        ray.add_argument(
            "--use_pixel_centers", action="store_true", default=False,
            help="add a half pixel while generating rays"
        )

        net = parser.add_argument_group("networks")
        net.add_argument("--netdepth", type=int, default=8, help="layers in network")
        net.add_argument("--netwidth", type=int, default=256, help="channels per layer")
        net.add_argument(
            "--netdepth_fine", type=int, default=8, help="layers in fine network"
        )
        net.add_argument(
            "--netwidth_fine", type=int, default=256, help="channels per layer in fine network"
        )

        # rendering options
        rendering = parser.add_argument_group("rendering")
        rendering.add_argument(
            "--num_coarse_samples", type=int, default=64, help="number of coarse samples per ray"
        )
        rendering.add_argument(
            "--num_fine_samples",
            type=int,
            default=0,
            help="number of additional fine samples per ray",
        )
        rendering.add_argument(
            "--perturb",
            type=float,
            default=1.0,
            help="set to 0. for no jitter, 1. for jitter",
        )
        rendering.add_argument(
            "--i_embed",
            type=int,
            default=0,
            help="set 0 for default positional encoding, -1 for none",
        )
        rendering.add_argument(
            "--multires",
            type=int,
            default=10,
            help="log2 of max freq for positional encoding (3D location)",
        )
        rendering.add_argument(
            "--multires_views",
            type=int,
            default=4,
            help="log2 of max freq for positional encoding (2D direction)",
        )
        rendering.add_argument(
            "--raw_noise_std",
            type=float,
            default=0.0,
            help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
        )

        # training options
        train = parser.add_argument_group("train")

        train.add_argument(
            "--lr_init", type=float, default=2.0e-4,
            help="initial learning rate"
        )
        train.add_argument(
            "--lr_final", type=float, default=2.5e-6, 
            help="the final learning rate"
        )
        train.add_argument(
            "--max_steps", type=int, default=1000000,
            help="the maximum number of steps"
        )
        train.add_argument(
            "--lr_delay_steps", type=int, default=2500, 
            help="learning rate delay step"
        )
        train.add_argument(
            "--lr_delay_mult", type=float, default=0.1,
            help="delay factor"
        )

        config = parser.add_argument_group("config")
        config.add_argument(
            "--config", is_config_file=True, help="config file path"
        )
        return parser.parse_args()

    def create_model(self):
        args = self.args
        embed_fn, input_ch = embedder.get_embedder(args.multires, args.i_embed)

        input_ch_views = 0
        embed_viewdirs_fn = None
        embed_viewdirs_fn, input_ch_views = embedder.get_embedder(
            args.multires_views, args.i_embed
        )
        output_ch = 5 if args.num_fine_samples > 0 else 4
        skips = [4]
        self.model = NeRF(
            D=args.netdepth,
            W=args.netwidth,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
        )

        if args.num_fine_samples > 0:
            self.model_fine = NeRF(
                D=args.netdepth_fine,
                W=args.netwidth_fine,
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
            netchunk=args.netchunk,
        )

        render_kwargs_train = {
            "network_query_fn": network_query_fn,
            "perturb": args.perturb,
            "num_fine_samples": args.num_fine_samples,
            "network_fine": self.model_fine,
            "num_coarse_samples": args.num_coarse_samples,
            "network_fn": self.model,
            "white_bkgd": args.white_bkgd,
            "raw_noise_std": args.raw_noise_std,
            "near": self.near,
            "far": self.far
        }

        # NDC only good for LLFF-style forward facing data
        if args.dataset_type != "llff" or args.no_ndc:
            render_kwargs_train["ndc"] = False
            render_kwargs_train["lindisp"] = args.lindisp

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
        return torch.optim.Adam(params=self.parameters(),
                                lr=self.args.lr_init,
                                betas=(0.9, 0.999))


    def forward_train(self, batch_rays):
        return self.forward_fun(rays=batch_rays, **self.render_kwargs_train)


    def forward_eval(self, batch_rays):
        return self.forward_fun(rays=batch_rays, **self.render_kwargs_test)


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

        self.log("train_psnr1", psnr, on_step=True, prog_bar=True, logger=True)
        self.log("train_psnr0", psnr0, on_step=True)
        self.log("train_loss", loss, on_step=True)
        return loss

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx,
        optimizer_closure, on_tpu, using_native_amp, using_lbfgs
    ):
        step = self.trainer.global_step
        if self.args.lr_delay_steps > 0:
            delay_rate = self.args.lr_delay_mult + (
                1 - self.args.lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(
                    step / self.args.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.

        t = np.clip(step / self.args.max_steps, 0, 1)
        scaled_lr = np.exp(
            np.log(self.args.lr_init) * (1 - t) +
            np.log(self.args.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        optimizer.step(closure=optimizer_closure)

    def render_rays(self, batch, batch_idx):
        ret = {}
        batch_rays = batch["ray"]
        rgb, disparity, acc, depth, extras = self.forward_eval(batch_rays)
        ret["rgb"] = rgb
        ret["depth"] = depth
        if "target" in batch:
            ret["target"] = batch["target"]
        return ret

    def validation_step(self, batch, batch_idx):
        return self.render_rays(batch, batch_idx)

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

            metrics.write_stats(os.path.join(self.logdir, "results.txt"), psnr, ssim, lpips)

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
            -1, self.img_size * 3)
        mse = torch.mean((target - rgbs)**2, dim=1)
        psnr = -10.0 * torch.log(mse) / np.log(10)
        psnr_mean = psnr.mean()
        self.log("val_psnr", psnr_mean, on_epoch=True, sync_dist=True)


class LitJaxNeRFBlender(LitJaxNeRF):
    def __init__(self, args):
        self.dataset = LitBlender(args)
        super(LitJaxNeRFBlender, self).__init__(args)


class LitJaxNeRFLLFF(LitJaxNeRF):
    def __init__(self, args):
        self.dataset = LitLLFF(args)
        super(LitJaxNeRFLLFF, self).__init__(args)
