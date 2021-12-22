from PIL import Image
import os
import numpy as np
import imageio


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def norm8b(x):
    x = (x - x.min()) / (x.max() - x.min())
    return to8b(x)


def store_image(dirpath, rgbs, depths):
    for (i, (rgb, depth)) in enumerate(zip(rgbs, depths)):
        imgname = f"image{str(i).zfill(3)}.jpg"
        depthname = f"depth{str(i).zfill(3)}.jpg"
        rgbimg = Image.fromarray(to8b(rgb))
        imgpath = os.path.join(dirpath, imgname)
        depthimg = Image.fromarray(norm8b(depth))
        depthpath = os.path.join(dirpath, depthname)
        rgbimg.save(imgpath)
        depthimg.save(depthpath)


def store_video(dirpath, rgbs, depths):    
    rgbimgs = [to8b(rgb) for rgb in rgbs]
    depthimgs = [norm8b(depth) for depth in depths]
    video_dir = os.path.join(dirpath, "videos")
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimwrite(os.path.join(video_dir, 'images.mp4'), rgbimgs, fps=30, quality=8)
    imageio.mimwrite(os.path.join(video_dir, 'depths.mp4'), depthimgs, fps=30, quality=8)

