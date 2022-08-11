import os

import imageio
import numpy as np
import torch
from PIL import Image


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def binary(x):
    x = np.round(x)
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def norm8b(x):
    x = (x - x.min()) / (x.max() - x.min())
    return to8b(x)


def store_image(dirpath, rgbs):
    for (i, rgb) in enumerate(rgbs):
        imgname = f"image{str(i).zfill(3)}.jpg"
        rgbimg = Image.fromarray(to8b(rgb.detach().cpu().numpy()))
        imgpath = os.path.join(dirpath, imgname)
        rgbimg.save(imgpath)


def store_depth(dirpath, depths):
    for (i, depth) in enumerate(depths):
        depthname = f"depth{str(i).zfill(3)}.jpg"
        disparity = torch.zeros_like(depth)
        disparity[torch.where(depth != 0)] = torch.log(
            (1 / (depth[torch.where(depth != 0)] + 1e-6))
        )
        img = norm8b(disparity.detach().cpu().numpy().repeat(3, axis=-1))
        depthimg = Image.fromarray(img)
        depthpath = os.path.join(dirpath, depthname)
        depthimg.save(depthpath)


def store_video(dirpath, rgbs):
    rgbimgs = [to8b(rgb.detach().cpu().numpy()) for rgb in rgbs]
    video_dir = os.path.join(dirpath, "videos")
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimwrite(os.path.join(video_dir, "images.mp4"), rgbimgs, fps=30, quality=8)


def store_mask(dirpath, masks):
    for (i, mask) in enumerate(masks):
        maskname = f"mask{str(i).zfill(3)}.jpg"
        maskimg = Image.fromarray(binary(mask.detach().cpu().numpy()))
        maskpath = os.path.join(dirpath, maskname)
        maskimg.save(maskpath)
