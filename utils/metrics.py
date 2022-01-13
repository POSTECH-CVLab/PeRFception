import numpy as np
import torch
import json

from piqa.ssim import SSIM
from piqa.lpips import LPIPS

reshape_2d = lambda x: x.reshape((x.shape[0], -1))
clip_0_1 = lambda x: torch.clip(x, 0, 1).detach()

@torch.no_grad()
def psnr(pred, gt, i_train, i_val, i_test):
    pred, gt = reshape_2d(pred), reshape_2d(gt)
    mse = np.mean((pred - gt) ** 2, axis=1)
    psnr = -10.0 * np.log(mse) / np.log(10)
    del pred, gt
    return {
        "name": "PSNR",
        "scene_wise": psnr, 
        "mean": psnr.mean(), 
        "train_mean": psnr[i_train].mean(), 
        "val_mean": psnr[i_val].mean(), 
        "test_mean": psnr[i_test].mean()
    }

@torch.no_grad()
def ssim(pred, gt, i_train, i_val, i_test):
    
    ssim_model = SSIM().cuda()
    pred = torch.from_numpy(pred)
    gt = torch.from_numpy(gt)
    pred_clip, gt_clip = clip_0_1(pred), clip_0_1(gt)
    ssim = []
    for i in range(len(pred_clip)):
        pred_i = pred_clip[None, i].permute((0, 3, 1, 2)).float().cuda()
        gt_i = gt_clip[None, i].permute((0, 3, 1, 2)).float().cuda()
        score = ssim_model(pred_i, gt_i).detach().cpu()
        del pred_i, gt_i
        ssim.append(score)
    ssim = torch.stack(ssim).numpy()
    del pred, gt, ssim_model, pred_clip, gt_clip
    return {
        "name": "SSIM",
        "scene_wise": ssim,
        "mean": ssim.mean(),
        "train_mean": ssim[i_train].mean(), 
        "val_mean": ssim[i_val].mean(), 
        "test_mean": ssim[i_test].mean()
    }

@torch.no_grad()
def lpips_a(pred, gt, i_train, i_val, i_test):
    lpips_model = LPIPS(network="alex").cuda()
    ret = lpips(lpips_model, pred, gt, i_train, i_val, i_test, "LPIPS-Alex")
    del lpips_model
    return ret

@torch.no_grad()
def lpips_v(pred, gt, i_train, i_val, i_test):
    lpips_model = LPIPS(network="vgg").cuda()
    ret = lpips(lpips_model, pred, gt, i_train, i_val, i_test, "LPIPS-VGG")
    del lpips_model
    return ret

def lpips(lpips_model, pred, gt, i_train, i_val, i_test, name):
    pred = torch.from_numpy(pred)
    gt = torch.from_numpy(gt)
    pred_clip, gt_clip = clip_0_1(pred), clip_0_1(gt)
    lpips = []
    for i in range(len(pred)): 
        pred_i = pred_clip[None, i].permute((0, 3, 1, 2)).float().cuda()
        gt_i = gt_clip[None, i].permute((0, 3, 1, 2)).float().cuda()
        score = lpips_model(pred_i, gt_i).detach().cpu()
        del pred_i, gt_i
        lpips.append(score)
    lpips = torch.stack(lpips).numpy()
    del pred, gt, pred_clip, gt_clip
    return {
        "name": name, 
        "scene_wise": lpips,
        "mean": lpips.mean(),
        "train_mean": lpips[i_train].mean(), 
        "val_mean": lpips[i_val].mean(), 
        "test_mean": lpips[i_test].mean()
    }


def write_stats(fpath, *stats):
    
    with open(fpath, "w") as fp:
        
        # MEAN
        fp.write("\nMEAN SCORE\n")
        for stat in stats:
            name, mean = stat["name"], stat["mean"]
            fp.write(f"   {name}: {mean}\n")

        # TRAIN MEAN
        fp.write("\nTRAIN MEAN SCORE\n")
        for stat in stats:
            name, mean = stat["name"], stat["train_mean"]
            fp.write(f"   {name}: {mean}\n")

        # VAL MEAN
        fp.write("\nVAL MEAN SCORE\n")
        for stat in stats:
            name, mean = stat["name"], stat["val_mean"]
            fp.write(f"   {name}: {mean}\n")

        # TEST MEAN
        fp.write("\nTEST MEAN SCORE\n")
        for stat in stats:
            name, mean = stat["name"], stat["test_mean"]
            fp.write(f"   {name}: {mean}\n")

        # SCENE-WISE SCORE
        fp.write("\nSCENE-WISE SCORE\n")
        N_img = len(stats[0]["scene_wise"])
        
        for i in range(N_img):
            img_name = f"image{str(i + 1).zfill(3)}\n"
            fp.write(f"    {img_name}")
            for stat in stats:
                name, score = stat["name"], stat["scene_wise"][i]
                fp.write(f"        {name}: {score}\n")

def write_stats_json(fpath, *stats):
    
    d = {}
    for stat in stats:
        d[stat["name"]] = {k : float(w) for (k, w) in stat.items() if k != "name" and k != "scene_wise"}

    with open(fpath, 'w') as fp:
        json.dump(d, fp)
