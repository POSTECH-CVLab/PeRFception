import numpy as np
import torch

from piqa.ssim import SSIM
from piqa.lpips import LPIPS

reshape_2d = lambda x: x.reshape((x.shape[0], -1))
clip_0_1 = lambda x: torch.clip(x, 0, 1)

def psnr(pred, gt, i_train, i_test):
    pred, gt = reshape_2d(pred), reshape_2d(gt)
    mse = np.mean((pred - gt) ** 2, axis=1)
    psnr = -10.0 * np.log(mse) / np.log(10)
    return {
        "name": "PSNR",
        "scene_wise": psnr, 
        "mean": psnr.mean(), 
        "train_mean": psnr[i_train].mean(), 
        "test_mean": psnr[i_test].mean()
    }

def ssim(pred, gt, i_train, i_test):
    
    ssim_model = SSIM().cuda()
    pred = torch.from_numpy(pred).to("cuda")
    gt = torch.from_numpy(gt).to("cuda")
    pred, gt = clip_0_1(pred), clip_0_1(gt)
    ssim = []
    for i in range(len(pred)):
        score = ssim_model(
            pred[None, i].permute((0, 3, 1, 2)), 
            gt[None, i].permute((0, 3, 1, 2))
        )
        ssim.append(score)
    ssim = torch.stack(ssim).cpu().numpy()
    return {
        "name": "SSIM",
        "scene_wise": ssim,
        "mean": ssim.mean(),
        "train_mean": ssim[i_train].mean(), 
        "test_mean": ssim[i_test].mean()
    }

def lpips_a(pred, gt, i_train, i_test):
    lpips_model = LPIPS(network="alex").cuda()
    return lpips(lpips_model, pred, gt, i_train, i_test, "LPIPS-Alex")

def lpips_v(pred, gt, i_train, i_test):
    lpips_model = LPIPS(network="vgg").cuda()
    return lpips(lpips_model, pred, gt, i_train, i_test, "LPIPS-VGG")

def lpips(lpips_model, pred, gt, i_train, i_test, name):
    pred = torch.from_numpy(pred).to("cuda")
    gt = torch.from_numpy(gt).to("cuda")
    pred, gt = clip_0_1(pred), clip_0_1(gt)
    lpips = []
    for i in range(len(pred)):
        score = lpips_model(
            pred[None, i].permute((0, 3, 1, 2)), 
            gt[None, i].permute((0, 3, 1, 2))
        )
        lpips.append(score)
    lpips = torch.stack(lpips).cpu().numpy()
    return {
        "name": name, 
        "scene_wise": lpips,
        "mean": lpips.mean(),
        "train_mean": lpips[i_train].mean(), 
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

