from PIL import Image
import os
import numpy as np
import imageio


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
        

def store_video(dirpath, rgbs):    
    rgbimgs = [to8b(rgb) for rgb in rgbs]
    video_dir = os.path.join(dirpath, "videos")
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimwrite(os.path.join(video_dir, 'images.mp4'), rgbimgs, fps=30, quality=8)

def store_mask(dirpath, masks):    
    for (i, mask) in enumerate(masks):
        maskname = f"mask{str(i).zfill(3)}.jpg"
        maskimg = Image.fromarray(binary(mask.detach().cpu().numpy()))
        maskpath = os.path.join(dirpath, maskname)
        maskimg.save(maskpath)
